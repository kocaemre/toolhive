// SPDX-FileCopyrightText: Copyright 2025 Stacklok, Inc.
// SPDX-License-Identifier: Apache-2.0

package tokenexchange

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"slices"
	"sync"
	"time"

	"github.com/go-jose/go-jose/v4"
	"github.com/go-jose/go-jose/v4/jwt"

	"github.com/stacklok/toolhive/pkg/oauthproto"
)

const (
	// jwksCacheTTL is the time-to-live for cached JWKS fetched from external issuers.
	jwksCacheTTL = 5 * time.Minute

	// httpTimeout is the timeout for HTTP requests to external OIDC endpoints.
	httpTimeout = 10 * time.Second

	// maxResponseBodySize is the maximum size of HTTP response bodies read from
	// external OIDC endpoints (1 MiB). This prevents resource exhaustion from
	// unexpectedly large responses.
	maxResponseBodySize = 1 << 20

	// maxJWKSKeys caps the number of keys accepted from an external JWKS to
	// prevent CPU amplification from a hostile endpoint serving many keys.
	maxJWKSKeys = 100

	// maxRedirects caps redirects followed when fetching external OIDC metadata.
	maxRedirects = 5

	// defaultActorClaim is the claim read to identify the client that
	// requested an external subject token, when a TrustedIssuer does not
	// configure ActorClaim. This matches Microsoft Entra v2 and many other
	// OIDC providers' convention for the authorized-party claim.
	defaultActorClaim = "azp"

	// externalClockSkewLeeway widens "nbf"/"iat"/"exp" acceptance for external
	// subject tokens to tolerate clock skew between the external IdP and
	// ToolHive. validateExternalToken independently rejects an expired
	// subject token by comparing its exp against time.Now() with zero
	// tolerance, so this leeway only protects against spurious
	// not-yet-valid/issued-in-the-future rejections — it never causes an
	// expired token to be accepted.
	externalClockSkewLeeway = 60 * time.Second
)

// actorClaimsNotInExtra are claims assignClaim drops or reroutes, so they
// never reach ValidatedClaims.Extra. Only "client_id" has an explicit
// fallback in resolveAllowedActor; configuring ActorClaim to any of these
// would make every external token look like it's missing the claim.
var actorClaimsNotInExtra = []string{
	"sub", "iss", "aud", "exp", "iat", "nbf", "jti", "name", "email", "scope", "may_act",
}

// Compile-time check that MultiIssuerTokenValidator implements SubjectTokenValidator.
var _ SubjectTokenValidator = (*MultiIssuerTokenValidator)(nil)

// TrustedIssuer configures an external OIDC issuer whose tokens are
// accepted as subject tokens during token exchange.
type TrustedIssuer struct {
	// IssuerURL is the expected "iss" claim value (exact match).
	IssuerURL string `json:"issuer_url" yaml:"issuer_url"`
	// ExpectedAudience is the expected "aud" claim value that must appear
	// in the token's audience list. Required; NewMultiIssuerTokenValidator
	// rejects any TrustedIssuer with an empty ExpectedAudience.
	ExpectedAudience string `json:"expected_audience" yaml:"expected_audience"`
	// JWKSURL is the URL to fetch the issuer's JSON Web Key Set from.
	// If empty, it is resolved via OIDC discovery at {IssuerURL}/.well-known/openid-configuration.
	JWKSURL string `json:"jwks_url,omitempty" yaml:"jwks_url,omitempty"`
	// ActorClaim names the claim that identifies the client that requested the
	// subject token from THIS EXTERNAL ISSUER, used for the AllowedActors consent
	// check below. Values are in the external issuer's client namespace — they are
	// NOT ToolHive client IDs, and listing a ToolHive client ID in AllowedActors
	// does not bind delegation to that client (see AllowedActors). Defaults to
	// "azp" when empty. Set to "appid" for Microsoft Entra v1 tokens, or "cid"
	// for Okta tokens. The special value "client_id" reads ValidatedClaims.ClientID
	// instead of Extra, since that claim is routed to a structured field rather
	// than left in Extra — it is still the external token's client_id claim, not
	// a ToolHive one.
	ActorClaim string `json:"actor_claim,omitempty" yaml:"actor_claim,omitempty"`
	// AllowedActors is the allowlist of ActorClaim values authorized to
	// exchange a subject token from this issuer, when the token does not
	// carry a "may_act" claim. Empty means only may_act-bearing tokens from
	// this issuer are accepted — every other token from it is rejected
	// (mirrors the empty-AllowedAudiences convention documented on
	// NewSelfIssuedTokenValidator).
	//
	// Accepted limitation (see #5989 and checkDelegationConsent's doc comment
	// in handler.go for the full rationale): an allowlisted actor satisfies
	// consent for ANY ToolHive confidential client holding the
	// token-exchange grant — every such client is delegation-equivalent, so
	// compromise of the weakest one suffices, and the allowlist itself gives
	// no per-client containment. Keeping this grant's client set minimal is
	// the operator's real control. Bounded by: the calling client must
	// already possess a valid subject token, and scope/audience narrowing
	// still applies to the exchanged result.
	AllowedActors []string `json:"allowed_actors,omitempty" yaml:"allowed_actors,omitempty"`
}

// MultiIssuerTokenValidator validates subject tokens from the authorization
// server itself or from configured external OIDC issuers.
//
// For self-issued tokens (where the "iss" claim matches selfIssuer), validation
// is delegated to the SelfIssuedTokenValidator. For tokens from trusted external
// issuers, the validator resolves the issuer's JWKS (via OIDC discovery if needed),
// verifies the JWT signature, and validates standard claims.
//
// A valid signature and audience alone would authorize ToolHive as a
// resource, not any particular client, as a delegate — a confused-deputy risk
// (CWE-863). An external token's "client_id" claim, when present, names a
// client in the EXTERNAL issuer's namespace, not a ToolHive client ID, so it
// cannot serve as the client_id-binding consent signal the self-issued path
// uses (checkDelegationConsent's client_id case in handler.go).
// validateExternalToken therefore requires one of two consent signals before
// returning successfully: a "may_act" claim (authoritative; enforced by the
// caller against the authenticated client), or the issuer's configured actor
// claim matching an entry in that issuer's AllowedActors — surfaced as
// ValidatedClaims.ExternalActor, which checkDelegationConsent must check
// before its client_id fallback.
type MultiIssuerTokenValidator struct {
	selfIssuer    string
	selfValidator *SelfIssuedTokenValidator
	issuers       map[string]*externalIssuerConfig
	httpClient    *http.Client

	// insecureSkipJWKSURLValidation disables HTTPS enforcement on discovered
	// JWKS URLs and relaxes the dial-time IP/scheme checks (so httptest servers
	// on loopback over HTTP are reachable). This MUST only be set for testing
	// with httptest servers.
	insecureSkipJWKSURLValidation bool
}

// externalIssuerConfig holds the configuration and cached state for an external
// OIDC issuer. The embedded TrustedIssuer is treated as immutable after
// construction: resolveAllowedActor reads TrustedIssuer.AllowedActors on every
// validation without holding mu, since mu protects JWKS state only. Mutating
// TrustedIssuer fields in place after NewMultiIssuerTokenValidator returns is a
// data race.
type externalIssuerConfig struct {
	TrustedIssuer

	mu      sync.Mutex
	jwksURL string              // resolved from OIDC discovery or preconfigured
	jwks    *jose.JSONWebKeySet // cached JWKS
	jwksExp time.Time           // when the cached JWKS expires
}

// NewMultiIssuerTokenValidator creates a validator that accepts tokens from the
// authorization server itself and from the provided list of trusted external issuers.
// Returns an error if selfValidator is nil, selfIssuer is empty, or any TrustedIssuer
// is invalid: empty IssuerURL or ExpectedAudience, an IssuerURL equal to selfIssuer
// or duplicated across entries, or an ActorClaim naming a claim assignClaim never
// leaves in Extra (see actorClaimsNotInExtra).
func NewMultiIssuerTokenValidator(
	selfValidator *SelfIssuedTokenValidator,
	selfIssuer string,
	trustedIssuers []TrustedIssuer,
) (*MultiIssuerTokenValidator, error) {
	if selfValidator == nil {
		return nil, errors.New("selfValidator must not be nil")
	}
	if selfIssuer == "" {
		return nil, errors.New("selfIssuer must not be empty")
	}

	issuers := make(map[string]*externalIssuerConfig, len(trustedIssuers))
	for _, ti := range trustedIssuers {
		if err := validateTrustedIssuer(ti, selfIssuer, issuers); err != nil {
			return nil, err
		}
		if len(ti.AllowedActors) == 0 {
			slog.Warn("Trusted issuer has no allowed actors configured; "+
				"only may_act-bearing subject tokens from it will be accepted",
				"issuer", ti.IssuerURL,
			)
		}

		// Clone AllowedActors so a caller mutating its original slice in place
		// (e.g. a future config reload) cannot race with the unsynchronized
		// reads in resolveAllowedActor, which is called on every validation
		// without holding externalIssuerConfig.mu (that mutex guards JWKS
		// state only).
		ti.AllowedActors = slices.Clone(ti.AllowedActors)

		issuers[ti.IssuerURL] = &externalIssuerConfig{
			TrustedIssuer: ti,
			jwksURL:       ti.JWKSURL,
		}
	}

	v := &MultiIssuerTokenValidator{
		selfIssuer:    selfIssuer,
		selfValidator: selfValidator,
		issuers:       issuers,
	}

	v.httpClient = &http.Client{
		Timeout: httpTimeout,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= maxRedirects {
				return errors.New("too many redirects")
			}
			// Re-validate the scheme of each redirect hop; the resolved IP
			// is checked in DialContext below.
			if !v.insecureSkipJWKSURLValidation && req.URL.Scheme != "https" {
				return fmt.Errorf("redirect to non-HTTPS URL: %q", req.URL.Scheme)
			}
			return nil
		},
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
				host, port, err := net.SplitHostPort(addr)
				if err != nil {
					return nil, err
				}
				ips, err := net.DefaultResolver.LookupIPAddr(ctx, host)
				if err != nil {
					return nil, err
				}
				for _, ipa := range ips {
					if !v.insecureSkipJWKSURLValidation && isDisallowedIP(ipa.IP) {
						return nil, fmt.Errorf("refusing to connect to disallowed address %s", ipa.IP)
					}
				}
				return (&net.Dialer{Timeout: 5 * time.Second}).DialContext(
					ctx, network, net.JoinHostPort(ips[0].String(), port))
			},
		},
	}

	return v, nil
}

// Validate parses the raw JWT to extract the issuer claim, then routes validation
// to either the self-issued validator or the appropriate external issuer validator.
// Returns an error if the issuer is not trusted.
func (v *MultiIssuerTokenValidator) Validate(ctx context.Context, rawToken string) (*ValidatedClaims, error) {
	// Parse the JWT without verification to peek at the issuer claim.
	// This is safe because we verify the signature in a subsequent step.
	issuer, err := peekIssuer(rawToken)
	if err != nil {
		return nil, fmt.Errorf("failed to determine token issuer: %w", err)
	}

	// Self-issued tokens are delegated to the existing validator.
	if issuer == v.selfIssuer {
		return v.selfValidator.Validate(ctx, rawToken)
	}

	// Look up the external issuer configuration.
	issuerConfig, ok := v.issuers[issuer]
	if !ok {
		return nil, fmt.Errorf("untrusted issuer: %q", issuer)
	}

	return v.validateExternalToken(ctx, rawToken, issuerConfig)
}

// validateExternalToken verifies a JWT from a trusted external issuer by
// fetching the issuer's JWKS (with caching) and validating the signature and claims.
func (v *MultiIssuerTokenValidator) validateExternalToken(
	ctx context.Context,
	rawToken string,
	issuerConfig *externalIssuerConfig,
) (*ValidatedClaims, error) {
	parsedToken, err := jwt.ParseSigned(rawToken, allowedSignatureAlgorithms)
	if err != nil {
		return nil, fmt.Errorf("subject token is not a valid JWT: %w", err)
	}

	jwks, err := v.resolveJWKS(ctx, issuerConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch JWKS for issuer %s: %w", issuerConfig.IssuerURL, err)
	}

	standardClaims, extraClaims, err := verifySignature(parsedToken, jwks)
	if err != nil {
		return nil, err
	}

	// Validate issuer and audience, tolerating externalClockSkewLeeway on
	// nbf/iat/exp. Expiry is enforced strictly (no leeway) below.
	expected := jwt.Expected{
		Issuer:      issuerConfig.IssuerURL,
		AnyAudience: jwt.Audience{issuerConfig.ExpectedAudience},
	}
	if err := standardClaims.ValidateWithLeeway(expected, externalClockSkewLeeway); err != nil {
		return nil, fmt.Errorf("subject token claims validation failed: %w", err)
	}

	// Subject is required for delegation.
	if standardClaims.Subject == "" {
		return nil, fmt.Errorf("subject token is missing required 'sub' claim")
	}

	// Expiry is required so the delegated token can be bounded by the subject
	// token's remaining lifetime.
	if standardClaims.Expiry == nil {
		return nil, errors.New("subject token is missing required 'exp' claim")
	}

	// Leeway above tolerates clock skew on nbf/iat, but an expired subject
	// token is never acceptable: it cannot bound the delegated token's
	// lifetime.
	if standardClaims.Expiry.Time().Before(time.Now()) {
		return nil, errors.New("subject token has expired")
	}

	// If may_act is present, it must be well-formed — see validateMayActShape.
	if err := validateMayActShape(extraClaims); err != nil {
		return nil, err
	}

	claims := buildValidatedClaims(standardClaims, extraClaims)

	// Delegation consent for the external path: a may_act claim is
	// authoritative and is enforced by the caller (checkDelegationConsent)
	// against the authenticated client, so nothing further is required here
	// — the AllowedActors allowlist below is skipped entirely whenever
	// MayAct is set. That means a trusted external issuer emitting may_act
	// controls consent directly: may_act.sub is later compared against a
	// ToolHive client ID by checkDelegationConsent, so an operator enabling
	// may_act on an external TrustedIssuer must ensure that claim is drawn
	// from ToolHive's own client namespace and cannot be influenced by an
	// untrusted party — it must not be treated as a value in the external
	// issuer's namespace the way the actor claim below is.
	//
	// Otherwise, the resolved actor claim must be present in this issuer's
	// AllowedActors — this is the consent signal for tokens without
	// may_act. Even when the resolved claim is "client_id" (ActorClaim:
	// "client_id"), it names a client in the external issuer's namespace,
	// not a ToolHive client ID, so it cannot be compared against the
	// authenticated ToolHive client the way ValidatedClaims.ClientID is in
	// the self-issued path.
	if claims.MayAct == nil {
		actor, err := resolveAllowedActor(issuerConfig, claims)
		if err != nil {
			return nil, err
		}
		claims.ExternalActor = actor
	}

	return claims, nil
}

// resolveJWKS returns the cached JWKS for an external issuer, fetching it if
// the cache is empty or expired. If the JWKS URL is not configured, it is first
// resolved via OIDC discovery.
func (v *MultiIssuerTokenValidator) resolveJWKS(
	ctx context.Context,
	issuerConfig *externalIssuerConfig,
) (*jose.JSONWebKeySet, error) {
	issuerConfig.mu.Lock()
	defer issuerConfig.mu.Unlock()
	// The lock is intentionally held across the network fetch so concurrent
	// validations of the same issuer don't trigger duplicate JWKS fetches.

	// Return cached JWKS if still valid.
	if issuerConfig.jwks != nil && time.Now().Before(issuerConfig.jwksExp) {
		return issuerConfig.jwks, nil
	}

	// Cache expired — clear the discovered URL so we re-discover on next fetch.
	// This handles the (rare) case where an issuer rotates its JWKS endpoint URL.
	// The explicitly configured JWKSURL (from TrustedIssuer) is preserved.
	if issuerConfig.JWKSURL == "" {
		issuerConfig.jwksURL = ""
	}

	// Discover the JWKS URL if not yet resolved.
	if issuerConfig.jwksURL == "" {
		jwksURL, err := v.discoverJWKSURL(ctx, issuerConfig.IssuerURL)
		if err != nil {
			return nil, fmt.Errorf("OIDC discovery failed for %s: %w", issuerConfig.IssuerURL, err)
		}
		issuerConfig.jwksURL = jwksURL
	}

	// Fetch and cache the JWKS.
	jwks, err := v.fetchJWKS(ctx, issuerConfig.jwksURL)
	if err != nil {
		return nil, err
	}

	issuerConfig.jwks = jwks
	issuerConfig.jwksExp = time.Now().Add(jwksCacheTTL)

	return jwks, nil
}

// peekIssuer parses a JWT without signature verification to extract the "iss" claim.
func peekIssuer(rawToken string) (string, error) {
	token, err := jwt.ParseSigned(rawToken, allowedSignatureAlgorithms)
	if err != nil {
		return "", fmt.Errorf("subject token is not a valid JWT: %w", err)
	}

	var claims jwt.Claims
	if err := token.UnsafeClaimsWithoutVerification(&claims); err != nil {
		return "", fmt.Errorf("failed to extract claims from subject token: %w", err)
	}

	if claims.Issuer == "" {
		return "", fmt.Errorf("subject token is missing 'iss' claim")
	}

	return claims.Issuer, nil
}

// discoverJWKSURL performs OIDC discovery to resolve the JWKS URL for an issuer.
// It fetches the OpenID Connect discovery document at {issuerURL}/.well-known/openid-configuration
// and extracts the jwks_uri field.
func (v *MultiIssuerTokenValidator) discoverJWKSURL(ctx context.Context, issuerURL string) (string, error) {
	discoveryURL := issuerURL + "/.well-known/openid-configuration"

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, discoveryURL, nil)
	if err != nil {
		return "", fmt.Errorf("failed to create discovery request: %w", err)
	}

	resp, err := v.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("discovery request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, resp.Body)
		return "", fmt.Errorf("discovery endpoint returned status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBodySize))
	if err != nil {
		return "", fmt.Errorf("failed to read discovery response: %w", err)
	}

	var doc oauthproto.OIDCDiscoveryDocument
	if err := json.Unmarshal(body, &doc); err != nil {
		return "", fmt.Errorf("failed to parse discovery document: %w", err)
	}

	if doc.Issuer != issuerURL {
		return "", fmt.Errorf("discovery document issuer %q does not match expected issuer %q", doc.Issuer, issuerURL)
	}

	if doc.JWKSURI == "" {
		return "", fmt.Errorf("discovery document missing 'jwks_uri'")
	}

	if !v.insecureSkipJWKSURLValidation {
		if err := validateJWKSURL(doc.JWKSURI); err != nil {
			return "", fmt.Errorf("discovered jwks_uri is invalid: %w", err)
		}
	}

	return doc.JWKSURI, nil
}

// validateJWKSURL checks that the JWKS URL uses HTTPS and is not a private/loopback address.
// This prevents SSRF attacks where a compromised discovery document points to internal services.
func validateJWKSURL(jwksURL string) error {
	u, err := url.Parse(jwksURL)
	if err != nil {
		return fmt.Errorf("invalid URL: %w", err)
	}

	if u.Scheme != "https" {
		return fmt.Errorf("must use HTTPS, got %q", u.Scheme)
	}

	host := u.Hostname()
	ip := net.ParseIP(host)
	if ip != nil && isDisallowedIP(ip) {
		return errors.New("must not point to a private or loopback address")
	}

	return nil
}

// isDisallowedIP reports whether an IP must not be dialed when fetching
// external OIDC metadata, blocking SSRF to internal/metadata addresses.
func isDisallowedIP(ip net.IP) bool {
	return ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() ||
		ip.IsLinkLocalMulticast() || ip.IsUnspecified()
}

// fetchJWKS fetches a JSON Web Key Set from the given URL.
func (v *MultiIssuerTokenValidator) fetchJWKS(ctx context.Context, jwksURL string) (*jose.JSONWebKeySet, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, jwksURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create JWKS request: %w", err)
	}

	resp, err := v.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("JWKS request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, resp.Body)
		return nil, fmt.Errorf("JWKS endpoint returned status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBodySize))
	if err != nil {
		return nil, fmt.Errorf("failed to read JWKS response: %w", err)
	}

	var jwks jose.JSONWebKeySet
	if err := json.Unmarshal(body, &jwks); err != nil {
		return nil, fmt.Errorf("failed to parse JWKS: %w", err)
	}

	if len(jwks.Keys) == 0 {
		return nil, errors.New("JWKS contains no keys")
	}
	if len(jwks.Keys) > maxJWKSKeys {
		return nil, fmt.Errorf("JWKS contains too many keys: %d (max %d)", len(jwks.Keys), maxJWKSKeys)
	}

	return &jwks, nil
}

// validateTrustedIssuer checks a single TrustedIssuer for structural validity
// before it is admitted into issuers: required fields, no collision with
// selfIssuer or an already-registered issuer, and an ActorClaim that
// resolveAllowedActor can actually read from Extra (or ClientID).
func validateTrustedIssuer(ti TrustedIssuer, selfIssuer string, issuers map[string]*externalIssuerConfig) error {
	if ti.IssuerURL == "" {
		return errors.New("trusted issuer: IssuerURL is required")
	}
	if ti.ExpectedAudience == "" {
		return fmt.Errorf("trusted issuer %q: ExpectedAudience is required", ti.IssuerURL)
	}
	if ti.IssuerURL == selfIssuer {
		return fmt.Errorf("trusted issuer %q: must not equal selfIssuer; "+
			"self-issued tokens are already handled separately", ti.IssuerURL)
	}
	if _, dup := issuers[ti.IssuerURL]; dup {
		return fmt.Errorf("trusted issuer %q: configured more than once", ti.IssuerURL)
	}
	if ti.ActorClaim != "" && slices.Contains(actorClaimsNotInExtra, ti.ActorClaim) {
		return fmt.Errorf(
			"trusted issuer %q: ActorClaim %q is not supported "+
				`(use "client_id" or a non-registered claim such as "azp", "appid", "cid")`,
			ti.IssuerURL, ti.ActorClaim)
	}
	return nil
}

// resolveAllowedActor resolves the issuer's configured actor claim from
// claims and checks it against issuerConfig.AllowedActors, returning the
// matched value on success. Called only when the subject token carries no
// may_act claim — see validateExternalToken.
func resolveAllowedActor(issuerConfig *externalIssuerConfig, claims *ValidatedClaims) (string, error) {
	claimName := issuerConfig.ActorClaim
	if claimName == "" {
		claimName = defaultActorClaim
	}

	// "client_id" is routed to ValidatedClaims.ClientID by assignClaim, so it
	// never appears in Extra; without this fallback a plausible operator
	// config (ActorClaim: "client_id") would silently reject all traffic.
	var raw any
	if claimName == "client_id" {
		raw = claims.ClientID
	} else {
		raw = claims.Extra[claimName]
	}

	actor, ok := raw.(string)
	if !ok || actor == "" {
		return "", fmt.Errorf(
			"subject token from issuer %q is missing or has an invalid %q claim required for delegation consent",
			issuerConfig.IssuerURL, claimName)
	}

	if !slices.Contains(issuerConfig.AllowedActors, actor) {
		return "", fmt.Errorf(
			"subject token from issuer %q names actor %q in claim %q, which is not in the allowed actors list",
			issuerConfig.IssuerURL, actor, claimName)
	}

	return actor, nil
}
