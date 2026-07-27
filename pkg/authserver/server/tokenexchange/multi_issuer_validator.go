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

	"github.com/stacklok/toolhive/pkg/networking"
	"github.com/stacklok/toolhive/pkg/oauthproto"
)

const (
	// jwksCacheTTL is the time-to-live for cached JWKS fetched from external issuers.
	jwksCacheTTL = 5 * time.Minute

	// jwksFailureBackoff bounds how often a failed JWKS fetch is retried for
	// the same issuer. resolveJWKS runs before the subject token's signature
	// is checked, so without this an authenticated client holding the
	// token-exchange grant could repeatedly abort the connection mid-fetch
	// and force a fresh discovery+JWKS round trip to the external IdP on
	// every attempt.
	jwksFailureBackoff = 30 * time.Second

	// httpTimeout is the timeout for HTTP requests to external OIDC endpoints.
	httpTimeout = 10 * time.Second

	// maxResponseBodySize is the maximum size of HTTP response bodies read from
	// external OIDC endpoints (1 MiB). This prevents resource exhaustion from
	// unexpectedly large responses.
	maxResponseBodySize = 1 << 20

	// maxJWKSKeys caps the number of keys accepted from an external JWKS to
	// prevent CPU amplification from a hostile endpoint serving many keys.
	maxJWKSKeys = 100

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
//
// This type is reused verbatim as the wire schema for
// authserver.RunConfig.TrustedIssuers (deliberately, to avoid a parallel
// type that drifts — see the go-style rule against that). Its JSON/YAML
// tags are therefore part of the serialized RunConfig, which is reflected
// into docs/server/swagger.*; adding, renaming, or retagging a field here is
// a schema change, not a purely internal one.
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
	// InsecureAllowHTTP permits plain-HTTP OIDC discovery and JWKS fetches
	// for THIS issuer only. Development and testing only — never set in
	// production. Does not relax the private-IP guard; see AllowPrivateIPs
	// for that. Deliberately per-issuer rather than a validator-wide or
	// self-issuer setting: this authorization server's own InsecureAllowHTTP
	// (e.g. for an in-cluster issuer) must not silently permit plaintext
	// discovery for every trusted external issuer too — a network attacker
	// who can intercept that traffic could substitute a JWKS and thereafter
	// forge subject tokens for that issuer's namespace.
	InsecureAllowHTTP bool `json:"insecure_allow_http,omitempty" yaml:"insecure_allow_http,omitempty"`
	// AllowPrivateIPs permits OIDC discovery and JWKS fetches for THIS
	// issuer to resolve to a private or loopback address. Use only when the
	// issuer is hosted inside the same cluster and has no public endpoint.
	AllowPrivateIPs bool `json:"allow_private_ips,omitempty" yaml:"allow_private_ips,omitempty"`
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
}

// externalIssuerConfig holds the configuration and cached state for an external
// OIDC issuer. The embedded TrustedIssuer is treated as immutable after
// construction: resolveAllowedActor reads TrustedIssuer.AllowedActors (and
// discoverJWKSURL/fetchJWKS read httpClient) on every validation without
// holding mu, since mu protects JWKS state only. Mutating TrustedIssuer
// fields in place after NewMultiIssuerTokenValidator returns is a data race.
type externalIssuerConfig struct {
	TrustedIssuer

	// httpClient is dedicated to this issuer, built once at construction
	// time from its own InsecureAllowHTTP/AllowPrivateIPs. A single
	// validator-wide client couldn't enforce per-issuer SSRF/transport
	// policy: http.Client.CheckRedirect and Transport.DialContext have no
	// way to know which issuer's fetch they are guarding.
	httpClient *http.Client

	mu      sync.Mutex
	jwksURL string              // resolved from OIDC discovery or preconfigured
	jwks    *jose.JSONWebKeySet // cached JWKS from the last successful fetch
	jwksErr error               // non-nil while jwksExp holds a failure-backoff deadline instead of a success expiry (see jwksFailureBackoff)
	jwksExp time.Time           // success cache expiry, or failure-backoff deadline when jwksErr != nil
}

// NewMultiIssuerTokenValidator creates a validator that accepts tokens from the
// authorization server itself and from the provided list of trusted external issuers.
// Returns an error if selfValidator is nil, selfIssuer is empty, any TrustedIssuer
// is invalid (empty IssuerURL or ExpectedAudience, an IssuerURL equal to selfIssuer
// or duplicated across entries, or an ActorClaim naming a claim assignClaim never
// leaves in Extra — see actorClaimsNotInExtra), or an issuer's dedicated HTTP
// client cannot be built.
//
// Each issuer gets its own *http.Client, built by
// networking.NewHttpClientBuilder from that issuer's own
// InsecureAllowHTTP/AllowPrivateIPs — never from a validator-wide flag,
// from this authorization server's own equivalent settings, or from any
// environment-variable bypass (see the comment where the client is built).
// See the doc comment on TrustedIssuer.InsecureAllowHTTP for why the
// per-issuer separation matters.
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

		// Deliberately networking.NewHttpClientBuilder(), not
		// NewHostScopedClientBuilder: that helper ORs
		// INSECURE_DISABLE_URL_VALIDATION and an auto-localhost exemption
		// into BOTH the HTTP-scheme and private-IP gates, so an unrelated
		// env var — or a trusted issuer that merely happens to be on
		// localhost — would silently widen AllowPrivateIPs regardless of
		// what the operator set. That defeats the point of splitting the
		// two flags per issuer. Passing InsecureAllowHTTP/AllowPrivateIPs
		// straight through keeps both gates independent and, for the
		// private-IP gate, env-independent (Build only installs the
		// dial-time private-IP guard when AllowPrivateIPs is false; that
		// guard itself never reads the environment).
		//
		// One residual: the built client's ValidatingTransport still skips
		// its HTTPS-scheme check when INSECURE_DISABLE_URL_VALIDATION is
		// set — that env read is baked into every builder-made client in
		// this repo. It is backstopped here: validateJWKSURL, called from
		// resolveJWKS before every fetch, enforces the scheme independently
		// of any environment variable, so it must stay there rather than
		// being treated as redundant with this client's own check.
		//
		// Unlike the sibling newHTTPClientForHost (upstream/oauth2.go),
		// keep-alives are disabled: that client dials one operator-configured
		// host repeatedly on a hot path, so it deliberately keeps them on.
		// This one dials jwks_uri — a host taken from an untrusted discovery
		// document — at most twice per jwksCacheTTL window, so it's the
		// "caller-varying host" case that comment's own doc says to revisit
		// for; there's no hot path here to trade the per-dial SSRF check away
		// for.
		httpClient, err := networking.NewHttpClientBuilder().
			WithInsecureAllowHTTP(ti.InsecureAllowHTTP).
			WithPrivateIPs(ti.AllowPrivateIPs).
			WithTimeout(httpTimeout).
			WithDisableKeepAlives(true).
			Build()
		if err != nil {
			return nil, fmt.Errorf("issuer_url %q: failed to build HTTP client: %w", ti.IssuerURL, err)
		}
		// Guard against a discovery/JWKS redirect hop landing on a
		// different, unvetted host — the same policy the transparent proxy
		// data path applies to a response derived from an untrusted remote
		// server (see SameHostRedirectPolicy's doc comment).
		httpClient.CheckRedirect = networking.SameHostRedirectPolicy()

		issuers[ti.IssuerURL] = &externalIssuerConfig{
			TrustedIssuer: ti,
			jwksURL:       ti.JWKSURL,
			httpClient:    httpClient,
		}
	}

	return &MultiIssuerTokenValidator{
		selfIssuer:    selfIssuer,
		selfValidator: selfValidator,
		issuers:       issuers,
	}, nil
}

// ValidateTrustedIssuers runs every structural check
// NewMultiIssuerTokenValidator performs on trustedIssuers — required fields,
// self-issuer collision, duplicate issuers, and ActorClaim reachability —
// without constructing a validator or any per-issuer HTTP client. Config
// validation calls this to fail before the live upstream DCR registration
// and storage creation that run between RunConfig.Validate and server
// construction; NewMultiIssuerTokenValidator repeats the same checks at
// server startup as defence in depth. Both route through validateTrustedIssuer,
// so the two can't drift out of sync.
func ValidateTrustedIssuers(trustedIssuers []TrustedIssuer, selfIssuer string) error {
	issuers := make(map[string]*externalIssuerConfig, len(trustedIssuers))
	for _, ti := range trustedIssuers {
		if err := validateTrustedIssuer(ti, selfIssuer, issuers); err != nil {
			return err
		}
		issuers[ti.IssuerURL] = &externalIssuerConfig{TrustedIssuer: ti}
	}
	return nil
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
// resolved via OIDC discovery. Before every fetch, the resolved URL (whether
// hand-configured or just discovered) is checked by validateJWKSURL against
// this issuer's own InsecureAllowHTTP/AllowPrivateIPs — the single point both
// paths pass through, so neither can bypass the HTTPS/SSRF guard.
//
// A failed fetch is cached too, for jwksFailureBackoff (see jwksErr): the
// caller (validateExternalToken) runs this before verifying the subject
// token's signature, so without backoff a client presenting a syntactically
// valid but unverifiable JWT naming this issuer could force a fresh
// discovery+fetch round trip on every attempt.
func (v *MultiIssuerTokenValidator) resolveJWKS(
	ctx context.Context,
	issuerConfig *externalIssuerConfig,
) (*jose.JSONWebKeySet, error) {
	issuerConfig.mu.Lock()
	defer issuerConfig.mu.Unlock()
	// The lock is intentionally held across the network fetch so concurrent
	// validations of the same issuer don't trigger duplicate JWKS fetches.

	now := time.Now()
	if now.Before(issuerConfig.jwksExp) {
		if issuerConfig.jwksErr != nil {
			return nil, issuerConfig.jwksErr
		}
		return issuerConfig.jwks, nil
	}

	// Detach from the caller's request context: net/http cancels ctx when
	// the client disconnects, and this fetch happens before the subject
	// token's signature is even checked, so an aborted connection must not
	// cut off a fetch other in-flight validations of this issuer are
	// waiting on (the lock above), nor let repeating the abort drive
	// unbounded outbound requests to the external IdP.
	fetchCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 2*httpTimeout)
	defer cancel()

	jwks, err := v.fetchAndValidateJWKS(fetchCtx, issuerConfig)
	if err != nil {
		issuerConfig.jwksErr = err
		issuerConfig.jwksExp = now.Add(jwksFailureBackoff)
		return nil, err
	}

	issuerConfig.jwks = jwks
	issuerConfig.jwksErr = nil
	issuerConfig.jwksExp = now.Add(jwksCacheTTL)
	return jwks, nil
}

// fetchAndValidateJWKS resolves issuerConfig.jwksURL — discovering it via
// OIDC first if not yet known — validates it, and fetches the JWKS it points
// to. Split out of resolveJWKS so the cache/backoff bookkeeping there has a
// single success/failure exit point to react to.
//
// Callers must hold issuerConfig.mu: this reads and writes issuerConfig.jwksURL.
func (v *MultiIssuerTokenValidator) fetchAndValidateJWKS(
	ctx context.Context,
	issuerConfig *externalIssuerConfig,
) (*jose.JSONWebKeySet, error) {
	// Cache expired — clear the discovered URL so we re-discover on next
	// fetch. This handles the (rare) case where an issuer rotates its JWKS
	// endpoint URL. The explicitly configured JWKSURL (from TrustedIssuer)
	// is preserved.
	if issuerConfig.JWKSURL == "" {
		issuerConfig.jwksURL = ""
	}

	// Discover the JWKS URL if not yet resolved.
	if issuerConfig.jwksURL == "" {
		jwksURL, err := v.discoverJWKSURL(ctx, issuerConfig)
		if err != nil {
			return nil, fmt.Errorf("OIDC discovery failed for %s: %w", issuerConfig.IssuerURL, err)
		}
		issuerConfig.jwksURL = jwksURL
	}

	// Validate the JWKS URL here, in the single choke point every fetch
	// passes through — whether it was hand-configured on TrustedIssuer or
	// just discovered above. A configured JWKSURL never reaches
	// discoverJWKSURL, so checking only there would leave hand-configured
	// URLs unvalidated.
	if err := validateJWKSURL(issuerConfig.jwksURL, issuerConfig.InsecureAllowHTTP, issuerConfig.AllowPrivateIPs); err != nil {
		return nil, fmt.Errorf("jwks_url for issuer %s is invalid: %w", issuerConfig.IssuerURL, err)
	}

	return v.fetchJWKS(ctx, issuerConfig)
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
// and extracts the jwks_uri field, using issuerConfig's own dedicated HTTP client.
func (*MultiIssuerTokenValidator) discoverJWKSURL(ctx context.Context, issuerConfig *externalIssuerConfig) (string, error) {
	discoveryURL := issuerConfig.IssuerURL + "/.well-known/openid-configuration"

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, discoveryURL, nil)
	if err != nil {
		return "", fmt.Errorf("failed to create discovery request: %w", err)
	}

	resp, err := issuerConfig.httpClient.Do(req)
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

	if doc.Issuer != issuerConfig.IssuerURL {
		return "", fmt.Errorf("discovery document issuer %q does not match expected issuer %q", doc.Issuer, issuerConfig.IssuerURL)
	}

	if doc.JWKSURI == "" {
		return "", fmt.Errorf("discovery document missing 'jwks_uri'")
	}

	// The returned URL is validated by the caller (resolveJWKS), which is
	// the single choke point covering both discovered and pre-configured
	// JWKS URLs.
	return doc.JWKSURI, nil
}

// validateJWKSURL checks that jwksURL uses HTTPS, unless insecureAllowHTTP
// permits plain HTTP, and is not a private/loopback address literal, unless
// allowPrivateIPs permits that. Both flags come from the specific
// TrustedIssuer being fetched (see resolveJWKS), never from a validator-wide
// or self-issuer setting. This prevents SSRF attacks where a compromised
// discovery document — or a hand-configured jwks_url — points to internal
// services.
func validateJWKSURL(jwksURL string, insecureAllowHTTP, allowPrivateIPs bool) error {
	u, err := url.Parse(jwksURL)
	if err != nil {
		return fmt.Errorf("invalid URL: %w", err)
	}

	if u.Scheme != "https" && !insecureAllowHTTP {
		return fmt.Errorf("must use HTTPS, got %q", u.Scheme)
	}

	host := u.Hostname()
	ip := net.ParseIP(host)
	if ip != nil && !allowPrivateIPs && networking.IsPrivateIP(ip) {
		return errors.New("must not point to a private or loopback address")
	}

	return nil
}

// fetchJWKS fetches a JSON Web Key Set from issuerConfig.jwksURL using its
// dedicated HTTP client.
func (*MultiIssuerTokenValidator) fetchJWKS(ctx context.Context, issuerConfig *externalIssuerConfig) (*jose.JSONWebKeySet, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, issuerConfig.jwksURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create JWKS request: %w", err)
	}

	resp, err := issuerConfig.httpClient.Do(req)
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
//
// Error messages name TrustedIssuer's wire keys (issuer_url,
// expected_audience, actor_claim), not its Go field names: TrustedIssuer is
// the serialized RunConfig.TrustedIssuers schema (see its doc comment), so an
// operator's YAML uses these keys and should see them echoed back, not Go
// identifiers they never wrote.
func validateTrustedIssuer(ti TrustedIssuer, selfIssuer string, issuers map[string]*externalIssuerConfig) error {
	if ti.IssuerURL == "" {
		return errors.New("issuer_url is required")
	}
	if ti.ExpectedAudience == "" {
		return fmt.Errorf("issuer_url %q: expected_audience is required", ti.IssuerURL)
	}
	if ti.IssuerURL == selfIssuer {
		return fmt.Errorf("issuer_url %q: must not equal the authorization server's own issuer; "+
			"self-issued tokens are already handled separately", ti.IssuerURL)
	}
	if _, dup := issuers[ti.IssuerURL]; dup {
		return fmt.Errorf("issuer_url %q: configured more than once", ti.IssuerURL)
	}
	if ti.ActorClaim != "" && slices.Contains(actorClaimsNotInExtra, ti.ActorClaim) {
		return fmt.Errorf(
			"issuer_url %q: actor_claim %q is not supported "+
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
