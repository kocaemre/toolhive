// SPDX-FileCopyrightText: Copyright 2025 Stacklok, Inc.
// SPDX-License-Identifier: Apache-2.0

package tokenexchange

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-jose/go-jose/v4"
	"github.com/go-jose/go-jose/v4/jwt"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	testExternalIssuer   = "https://keycloak.example.com/realms/test"
	testExternalAudience = "toolhive-authserver"
)

// startJWKSServer creates a test HTTP server that serves a JWKS endpoint.
// The returned server must be closed by the caller.
func startJWKSServer(t *testing.T, tj *testJWKS) *httptest.Server {
	t.Helper()

	mux := http.NewServeMux()
	mux.HandleFunc("/jwks", func(w http.ResponseWriter, _ *http.Request) {
		// Serve only the public keys.
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(tj.publicJWKS())
	})

	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

// startDiscoveryServer creates a test HTTP server that serves both OIDC discovery
// and JWKS endpoints, simulating an external OIDC provider.
func startDiscoveryServer(t *testing.T, tj *testJWKS) *httptest.Server {
	t.Helper()

	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, r *http.Request) {
		// The issuer and jwks_uri must use the test server's own base URL,
		// which we don't know until the server starts. We use the Host header
		// to construct them. The issuer must match the requested issuer URL
		// (which the test configures as the discovery server's own URL).
		base := "http://" + r.Host
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{
			"issuer":   base,
			"jwks_uri": base + "/jwks",
		})
	})
	mux.HandleFunc("/jwks", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(tj.publicJWKS())
	})

	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

// newMultiValidator creates a MultiIssuerTokenValidator configured for testing.
// The external JWKS URL is pre-resolved (no discovery needed) unless jwksURL is empty.
func newMultiValidator(
	t *testing.T,
	selfJWKS *testJWKS,
	trustedIssuers []TrustedIssuer,
) *MultiIssuerTokenValidator {
	t.Helper()

	selfValidator, err := NewSelfIssuedTokenValidator(selfJWKS.publicJWKS(), testIssuer, []string{testIssuer})
	require.NoError(t, err)

	// Copy rather than mutate the caller's slice: give every test issuer
	// permissive flags so its httptest discovery/JWKS server, which runs on
	// loopback over plain HTTP, is reachable.
	issuers := make([]TrustedIssuer, len(trustedIssuers))
	for i, ti := range trustedIssuers {
		ti.InsecureAllowHTTP = true
		ti.AllowPrivateIPs = true
		issuers[i] = ti
	}

	v, err := NewMultiIssuerTokenValidator(selfValidator, testIssuer, issuers)
	require.NoError(t, err)
	return v
}

// externalClaims returns standard JWT claims for a token issued by the external issuer.
func externalClaims() jwt.Claims {
	now := time.Now()
	return jwt.Claims{
		Subject:   "ext-user-456",
		Issuer:    testExternalIssuer,
		Audience:  jwt.Audience{testExternalAudience},
		Expiry:    jwt.NewNumericDate(now.Add(time.Hour)),
		IssuedAt:  jwt.NewNumericDate(now),
		NotBefore: jwt.NewNumericDate(now.Add(-time.Minute)),
		ID:        "jti-ext-789",
	}
}

func TestMultiIssuerTokenValidator_Validate(t *testing.T) {
	t.Parallel()

	selfJWKS := newTestJWKS(t)
	externalJWKS := newTestJWKS(t)
	jwksServer := startJWKSServer(t, externalJWKS)

	tests := []struct {
		name           string
		trustedIssuers []TrustedIssuer
		token          func(t *testing.T) string
		wantErr        bool
		errContains    string
		check          func(t *testing.T, vc *ValidatedClaims)
	}{
		{
			name: "self-issued token routes to self validator",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
			}},
			token: func(t *testing.T) string {
				t.Helper()
				return selfJWKS.signToken(t, validClaims(), validExtraClaims())
			},
			check: func(t *testing.T, vc *ValidatedClaims) {
				t.Helper()
				assert.Equal(t, "user-123", vc.Subject)
				assert.Equal(t, testIssuer, vc.Issuer)
				assert.Equal(t, []string{testIssuer}, vc.Audience)
				assert.Equal(t, "Test User", vc.Name)
				assert.Equal(t, "test@example.com", vc.Email)
			},
		},
		{
			name: "external token accepted",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
				AllowedActors:    []string{"ext-agent"},
			}},
			token: func(t *testing.T) string {
				t.Helper()
				return externalJWKS.signToken(t, externalClaims(), map[string]interface{}{
					"name":  "External User",
					"email": "ext@keycloak.example.com",
					"azp":   "ext-agent",
				})
			},
			check: func(t *testing.T, vc *ValidatedClaims) {
				t.Helper()
				assert.Equal(t, "ext-user-456", vc.Subject)
				assert.Equal(t, testExternalIssuer, vc.Issuer)
				assert.Equal(t, []string{testExternalAudience}, vc.Audience)
				assert.Equal(t, "jti-ext-789", vc.JWTID)
				assert.Equal(t, "External User", vc.Name)
				assert.Equal(t, "ext@keycloak.example.com", vc.Email)
				assert.False(t, vc.Expiry.IsZero())
				assert.False(t, vc.IssuedAt.IsZero())
				assert.Equal(t, "ext-agent", vc.ExternalActor, "actor claim matched via default azp")
			},
		},
		{
			name: "external token custom actor claim appid accepted",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
				ActorClaim:       "appid",
				AllowedActors:    []string{"ext-agent"},
			}},
			token: func(t *testing.T) string {
				t.Helper()
				return externalJWKS.signToken(t, externalClaims(), map[string]any{"appid": "ext-agent"})
			},
			check: func(t *testing.T, vc *ValidatedClaims) {
				t.Helper()
				assert.Equal(t, "ext-agent", vc.ExternalActor)
			},
		},
		{
			name: "external token custom actor claim cid accepted",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
				ActorClaim:       "cid",
				AllowedActors:    []string{"ext-agent"},
			}},
			token: func(t *testing.T) string {
				t.Helper()
				return externalJWKS.signToken(t, externalClaims(), map[string]any{"cid": "ext-agent"})
			},
			check: func(t *testing.T, vc *ValidatedClaims) {
				t.Helper()
				assert.Equal(t, "ext-agent", vc.ExternalActor)
			},
		},
		{
			name: "external token actor claim client_id resolves from ClientID field",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
				ActorClaim:       "client_id",
				AllowedActors:    []string{"ext-agent"},
			}},
			token: func(t *testing.T) string {
				t.Helper()
				// "client_id" is routed to ValidatedClaims.ClientID by assignClaim, so
				// it never lands in Extra — resolveAllowedActor must fall back to
				// reading the structured field instead.
				return externalJWKS.signToken(t, externalClaims(), map[string]any{"client_id": "ext-agent"})
			},
			check: func(t *testing.T, vc *ValidatedClaims) {
				t.Helper()
				assert.Equal(t, "ext-agent", vc.ClientID)
				assert.Equal(t, "ext-agent", vc.ExternalActor)
			},
		},
		{
			name: "external token actor not in allowed actors rejected",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
				AllowedActors:    []string{"ext-agent"},
			}},
			token: func(t *testing.T) string {
				t.Helper()
				return externalJWKS.signToken(t, externalClaims(), map[string]any{"azp": "some-other-client"})
			},
			wantErr:     true,
			errContains: "not in the allowed actors list",
		},
		{
			name: "external token missing actor claim rejected",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
				AllowedActors:    []string{"ext-agent"},
			}},
			token: func(t *testing.T) string {
				t.Helper()
				return externalJWKS.signToken(t, externalClaims(), nil)
			},
			wantErr:     true,
			errContains: "required for delegation consent",
		},
		{
			name: "external token actor claim is a number rejected",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
				AllowedActors:    []string{"ext-agent"},
			}},
			token: func(t *testing.T) string {
				t.Helper()
				return externalJWKS.signToken(t, externalClaims(), map[string]any{"azp": 123})
			},
			wantErr:     true,
			errContains: "required for delegation consent",
		},
		{
			name: "external token actor claim is a bool rejected",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
				AllowedActors:    []string{"ext-agent"},
			}},
			token: func(t *testing.T) string {
				t.Helper()
				return externalJWKS.signToken(t, externalClaims(), map[string]any{"azp": true})
			},
			wantErr:     true,
			errContains: "required for delegation consent",
		},
		{
			name: "external token actor claim is an array rejected",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
				AllowedActors:    []string{"ext-agent"},
			}},
			token: func(t *testing.T) string {
				t.Helper()
				return externalJWKS.signToken(t, externalClaims(), map[string]any{"azp": []string{"ext-agent"}})
			},
			wantErr:     true,
			errContains: "required for delegation consent",
		},
		{
			name: "external token actor claim is a nested object rejected",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
				AllowedActors:    []string{"ext-agent"},
			}},
			token: func(t *testing.T) string {
				t.Helper()
				return externalJWKS.signToken(t, externalClaims(),
					map[string]any{"azp": map[string]any{"id": "ext-agent"}})
			},
			wantErr:     true,
			errContains: "required for delegation consent",
		},
		{
			name: "external token actor claim empty string rejected",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
				AllowedActors:    []string{"ext-agent"},
			}},
			token: func(t *testing.T) string {
				t.Helper()
				return externalJWKS.signToken(t, externalClaims(), map[string]any{"azp": ""})
			},
			wantErr:     true,
			errContains: "required for delegation consent",
		},
		{
			name: "external token no may_act and empty allowed actors rejected",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
				// AllowedActors intentionally empty.
			}},
			token: func(t *testing.T) string {
				t.Helper()
				return externalJWKS.signToken(t, externalClaims(), map[string]any{"azp": "ext-agent"})
			},
			wantErr:     true,
			errContains: "not in the allowed actors list",
		},
		{
			name: "external token with may_act and empty allowed actors accepted",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
				// AllowedActors intentionally empty: may_act is authoritative and
				// skips the allowlist entirely.
			}},
			token: func(t *testing.T) string {
				t.Helper()
				return externalJWKS.signToken(t, externalClaims(),
					map[string]any{"may_act": map[string]any{"sub": "some-toolhive-client"}})
			},
			check: func(t *testing.T, vc *ValidatedClaims) {
				t.Helper()
				require.NotNil(t, vc.MayAct)
				assert.Equal(t, "some-toolhive-client", vc.MayAct.Sub)
				assert.Empty(t, vc.ExternalActor, "allowlist is skipped whenever may_act is present")
			},
		},
		{
			name: "external token may_act wins even when azp is not allowlisted",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
				AllowedActors:    []string{"someone-else"},
			}},
			token: func(t *testing.T) string {
				t.Helper()
				return externalJWKS.signToken(t, externalClaims(), map[string]any{
					"azp":     "not-in-the-allowlist",
					"may_act": map[string]any{"sub": "some-toolhive-client"},
				})
			},
			check: func(t *testing.T, vc *ValidatedClaims) {
				t.Helper()
				require.NotNil(t, vc.MayAct)
				assert.Empty(t, vc.ExternalActor)
			},
		},
		{
			name: "external token malformed may_act — JSON string, not object — rejected outright",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
				AllowedActors:    []string{"ext-agent"},
			}},
			token: func(t *testing.T) string {
				t.Helper()
				// azp is allowlisted, so a regression that fell through to the
				// allowlist instead of rejecting would wrongly accept this.
				return externalJWKS.signToken(t, externalClaims(), map[string]any{
					"azp":     "ext-agent",
					"may_act": "agent-a",
				})
			},
			wantErr:     true,
			errContains: "malformed 'may_act' claim",
		},
		{
			name: "external token malformed may_act — non-string sub — rejected outright",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
				AllowedActors:    []string{"ext-agent"},
			}},
			token: func(t *testing.T) string {
				t.Helper()
				return externalJWKS.signToken(t, externalClaims(), map[string]any{
					"azp":     "ext-agent",
					"may_act": map[string]any{"sub": 123},
				})
			},
			wantErr:     true,
			errContains: "malformed 'may_act' claim",
		},
		{
			name: "external token malformed may_act — empty sub — rejected outright",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
				AllowedActors:    []string{"ext-agent"},
			}},
			token: func(t *testing.T) string {
				t.Helper()
				return externalJWKS.signToken(t, externalClaims(), map[string]any{
					"azp":     "ext-agent",
					"may_act": map[string]any{"sub": ""},
				})
			},
			wantErr:     true,
			errContains: "malformed 'may_act' claim",
		},
		{
			name: "external token malformed may_act — array sub — rejected outright",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
				AllowedActors:    []string{"ext-agent"},
			}},
			token: func(t *testing.T) string {
				t.Helper()
				return externalJWKS.signToken(t, externalClaims(), map[string]any{
					"azp":     "ext-agent",
					"may_act": map[string]any{"sub": []string{"agent-a"}},
				})
			},
			wantErr:     true,
			errContains: "malformed 'may_act' claim",
		},
		{
			name: "external token malformed may_act — object with no sub key — rejected outright",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
				AllowedActors:    []string{"ext-agent"},
			}},
			token: func(t *testing.T) string {
				t.Helper()
				return externalJWKS.signToken(t, externalClaims(), map[string]any{
					"azp":     "ext-agent",
					"may_act": map[string]any{"subject": "agent-a"},
				})
			},
			wantErr:     true,
			errContains: "malformed 'may_act' claim",
		},
		{
			name: "external token wrong audience",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
			}},
			token: func(t *testing.T) string {
				t.Helper()
				claims := externalClaims()
				claims.Audience = jwt.Audience{"wrong-audience"}
				return externalJWKS.signToken(t, claims, nil)
			},
			wantErr:     true,
			errContains: "claims validation failed",
		},
		{
			name: "unknown issuer rejected",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
			}},
			token: func(t *testing.T) string {
				t.Helper()
				claims := externalClaims()
				claims.Issuer = "https://evil.example.com"
				return externalJWKS.signToken(t, claims, nil)
			},
			wantErr:     true,
			errContains: "untrusted issuer",
		},
		{
			name: "external token bad signature",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
			}},
			token: func(t *testing.T) string {
				t.Helper()
				// Sign with a different key than the one the JWKS server serves.
				wrongJWKS := newTestJWKS(t)
				return wrongJWKS.signToken(t, externalClaims(), nil)
			},
			wantErr:     true,
			errContains: "signature verification failed",
		},
		{
			name: "self-issued token signed by external key fails",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
			}},
			token: func(t *testing.T) string {
				t.Helper()
				// Token claims say iss=self, but signed by the external key.
				// Routes to self validator, which rejects the signature.
				return externalJWKS.signToken(t, validClaims(), nil)
			},
			wantErr:     true,
			errContains: "signature verification failed",
		},
		{
			name: "external token missing subject",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
			}},
			token: func(t *testing.T) string {
				t.Helper()
				claims := externalClaims()
				claims.Subject = ""
				return externalJWKS.signToken(t, claims, nil)
			},
			wantErr:     true,
			errContains: "missing required 'sub' claim",
		},
		{
			name: "external token missing exp claim",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
				AllowedActors:    []string{"ext-agent"},
			}},
			token: func(t *testing.T) string {
				t.Helper()
				claims := externalClaims()
				claims.Expiry = nil
				return externalJWKS.signToken(t, claims, map[string]any{"azp": "ext-agent"})
			},
			wantErr:     true,
			errContains: "missing required 'exp' claim",
		},
		{
			// Distinct from "malformed token" below: that hits the JWT parse
			// failure inside peekIssuer, this hits its separate empty-issuer
			// check on an otherwise well-formed token. Both wrap to the same
			// outer "failed to determine token issuer" message, so assert on
			// the distinct inner text instead.
			name: "token missing iss claim",
			token: func(t *testing.T) string {
				t.Helper()
				claims := externalClaims()
				claims.Issuer = ""
				return externalJWKS.signToken(t, claims, nil)
			},
			wantErr:     true,
			errContains: "missing 'iss' claim",
		},
		{
			// Proves the allowlist is skipped (not just "would have failed
			// anyway") whenever may_act is present: both azp and
			// AllowedActors would satisfy resolveAllowedActor if it ran, so
			// a regression that calls it unconditionally and only gates the
			// error on MayAct==nil would still populate ExternalActor here.
			name: "external token may_act present alongside an allowlisted azp still yields empty ExternalActor",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
				AllowedActors:    []string{"ext-agent"},
			}},
			token: func(t *testing.T) string {
				t.Helper()
				return externalJWKS.signToken(t, externalClaims(), map[string]any{
					"azp":     "ext-agent",
					"may_act": map[string]any{"sub": "some-toolhive-client"},
				})
			},
			check: func(t *testing.T, vc *ValidatedClaims) {
				t.Helper()
				require.NotNil(t, vc.MayAct)
				assert.Empty(t, vc.ExternalActor, "allowlist must be skipped entirely when may_act is present")
			},
		},
		{
			name: "external token expired",
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          jwksServer.URL + "/jwks",
			}},
			token: func(t *testing.T) string {
				t.Helper()
				claims := externalClaims()
				claims.Expiry = jwt.NewNumericDate(time.Now().Add(-time.Hour))
				claims.IssuedAt = jwt.NewNumericDate(time.Now().Add(-2 * time.Hour))
				return externalJWKS.signToken(t, claims, nil)
			},
			wantErr:     true,
			errContains: "claims validation failed",
		},
		{
			name:           "malformed token",
			trustedIssuers: nil,
			token: func(_ *testing.T) string {
				return "not-a-jwt"
			},
			wantErr:     true,
			errContains: "failed to determine token issuer",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			validator := newMultiValidator(t, selfJWKS, tt.trustedIssuers)
			rawToken := tt.token(t)

			result, err := validator.Validate(context.Background(), rawToken)

			if tt.wantErr {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.errContains)
				assert.Nil(t, result)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, result)
			if tt.check != nil {
				tt.check(t, result)
			}
		})
	}
}

func TestMultiIssuerTokenValidator_OIDCDiscovery(t *testing.T) {
	t.Parallel()

	selfJWKS := newTestJWKS(t)
	externalJWKS := newTestJWKS(t)
	discoveryServer := startDiscoveryServer(t, externalJWKS)

	// Configure the trusted issuer WITHOUT a JWKS URL, forcing OIDC discovery.
	// The discovery server's URL is used as the issuer URL so that the
	// /.well-known/openid-configuration endpoint is reachable.
	trustedIssuers := []TrustedIssuer{{
		IssuerURL:        discoveryServer.URL,
		ExpectedAudience: testExternalAudience,
		AllowedActors:    []string{"ext-agent"},
		// JWKSURL intentionally left empty to trigger discovery.
	}}

	validator := newMultiValidator(t, selfJWKS, trustedIssuers)

	// Sign a token with the external key, using the discovery server's URL as issuer.
	claims := jwt.Claims{
		Subject:   "discovered-user",
		Issuer:    discoveryServer.URL,
		Audience:  jwt.Audience{testExternalAudience},
		Expiry:    jwt.NewNumericDate(time.Now().Add(time.Hour)),
		IssuedAt:  jwt.NewNumericDate(time.Now()),
		NotBefore: jwt.NewNumericDate(time.Now().Add(-time.Minute)),
		ID:        "jti-disc-001",
	}
	rawToken := externalJWKS.signToken(t, claims, map[string]any{"azp": "ext-agent"})

	result, err := validator.Validate(context.Background(), rawToken)
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, "discovered-user", result.Subject)
	assert.Equal(t, discoveryServer.URL, result.Issuer)
}

func TestMultiIssuerTokenValidator_JWKSCaching(t *testing.T) {
	t.Parallel()

	selfJWKS := newTestJWKS(t)
	externalJWKS := newTestJWKS(t)

	// Track how many times the JWKS endpoint is hit.
	var fetchCount atomic.Int32
	mux := http.NewServeMux()
	mux.HandleFunc("/jwks", func(w http.ResponseWriter, _ *http.Request) {
		fetchCount.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(externalJWKS.publicJWKS())
	})
	jwksServer := httptest.NewServer(mux)
	t.Cleanup(jwksServer.Close)

	trustedIssuers := []TrustedIssuer{{
		IssuerURL:        testExternalIssuer,
		ExpectedAudience: testExternalAudience,
		JWKSURL:          jwksServer.URL + "/jwks",
		AllowedActors:    []string{"ext-agent"},
	}}

	validator := newMultiValidator(t, selfJWKS, trustedIssuers)

	// Validate two tokens — the JWKS should be fetched only once (cached).
	for i := range 2 {
		claims := externalClaims()
		claims.ID = fmt.Sprintf("jti-cache-%d", i)
		rawToken := externalJWKS.signToken(t, claims, map[string]any{"azp": "ext-agent"})

		result, err := validator.Validate(context.Background(), rawToken)
		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Equal(t, "ext-user-456", result.Subject)
	}

	assert.Equal(t, int32(1), fetchCount.Load(), "JWKS should be fetched only once due to caching")
}

func TestNewMultiIssuerTokenValidator_Validation(t *testing.T) {
	t.Parallel()

	selfJWKS := newTestJWKS(t)
	selfValidator, err := NewSelfIssuedTokenValidator(selfJWKS.publicJWKS(), testIssuer, []string{testIssuer})
	require.NoError(t, err)

	tests := []struct {
		name           string
		selfValidator  *SelfIssuedTokenValidator
		selfIssuer     string
		trustedIssuers []TrustedIssuer
		errContains    string
	}{
		{
			name:        "nil selfValidator rejected",
			selfIssuer:  testIssuer,
			errContains: "selfValidator must not be nil",
		},
		{
			name:          "empty selfIssuer rejected",
			selfValidator: selfValidator,
			selfIssuer:    "",
			errContains:   "selfIssuer must not be empty",
		},
		{
			name:          "empty IssuerURL rejected",
			selfValidator: selfValidator,
			selfIssuer:    testIssuer,
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        "",
				ExpectedAudience: testExternalAudience,
			}},
			errContains: "issuer_url is required",
		},
		{
			name:          "empty ExpectedAudience rejected",
			selfValidator: selfValidator,
			selfIssuer:    testIssuer,
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: "",
			}},
			errContains: "expected_audience is required",
		},
		{
			name:          "IssuerURL equal to selfIssuer rejected",
			selfValidator: selfValidator,
			selfIssuer:    testIssuer,
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testIssuer,
				ExpectedAudience: testExternalAudience,
			}},
			errContains: "must not equal the authorization server's own issuer",
		},
		{
			name:          "duplicate IssuerURL rejected",
			selfValidator: selfValidator,
			selfIssuer:    testIssuer,
			trustedIssuers: []TrustedIssuer{
				{IssuerURL: testExternalIssuer, ExpectedAudience: testExternalAudience},
				{IssuerURL: testExternalIssuer, ExpectedAudience: testExternalAudience},
			},
			errContains: "configured more than once",
		},
		{
			name:          `ActorClaim "sub" rejected — assignClaim never leaves it in Extra`,
			selfValidator: selfValidator,
			selfIssuer:    testIssuer,
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				ActorClaim:       "sub",
			}},
			errContains: "actor_claim",
		},
		{
			name:          `ActorClaim "scope" rejected — rerouted to a structured field`,
			selfValidator: selfValidator,
			selfIssuer:    testIssuer,
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				ActorClaim:       "scope",
			}},
			errContains: "actor_claim",
		},
		{
			name:          `ActorClaim "may_act" rejected — rerouted to a structured field`,
			selfValidator: selfValidator,
			selfIssuer:    testIssuer,
			trustedIssuers: []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				ActorClaim:       "may_act",
			}},
			errContains: "actor_claim",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			v, err := NewMultiIssuerTokenValidator(tt.selfValidator, tt.selfIssuer, tt.trustedIssuers)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.errContains)
			assert.Nil(t, v)
		})
	}
}

func TestNewMultiIssuerTokenValidator_EmptyAllowedActorsAccepted(t *testing.T) {
	t.Parallel()

	selfJWKS := newTestJWKS(t)
	selfValidator, err := NewSelfIssuedTokenValidator(selfJWKS.publicJWKS(), testIssuer, []string{testIssuer})
	require.NoError(t, err)

	// Empty AllowedActors must be accepted by the constructor: a may_act-only
	// issuer (no allowlisted actors at all) is a legitimate configuration.
	v, err := NewMultiIssuerTokenValidator(selfValidator, testIssuer, []TrustedIssuer{{
		IssuerURL:        testExternalIssuer,
		ExpectedAudience: testExternalAudience,
	}})
	require.NoError(t, err)
	assert.NotNil(t, v)
}

func TestNewMultiIssuerTokenValidator_ClonesAllowedActors(t *testing.T) {
	t.Parallel()

	selfJWKS := newTestJWKS(t)
	externalJWKS := newTestJWKS(t)
	jwksServer := startJWKSServer(t, externalJWKS)

	allowedActors := []string{"ext-agent"}
	trustedIssuers := []TrustedIssuer{{
		IssuerURL:        testExternalIssuer,
		ExpectedAudience: testExternalAudience,
		JWKSURL:          jwksServer.URL + "/jwks",
		AllowedActors:    allowedActors,
	}}
	validator := newMultiValidator(t, selfJWKS, trustedIssuers)

	// Mutate the caller's slice after construction. If the constructor didn't
	// clone it, this would change what the validator accepts.
	allowedActors[0] = "someone-else"

	rawToken := externalJWKS.signToken(t, externalClaims(), map[string]any{"azp": "ext-agent"})
	result, err := validator.Validate(context.Background(), rawToken)
	require.NoError(t, err, "post-construction mutation of the caller's slice must not affect validation")
	require.NotNil(t, result)
	assert.Equal(t, "ext-agent", result.ExternalActor)
}

// TestMultiIssuerTokenValidator_ClockSkewLeeway exercises externalClockSkewLeeway
// (60s): nbf/iat tolerate it, but exp is enforced strictly by an independent
// check in validateExternalToken, since go-jose's leeway would otherwise widen
// exp too and let a genuinely expired subject token through.
func TestMultiIssuerTokenValidator_ClockSkewLeeway(t *testing.T) {
	t.Parallel()

	selfJWKS := newTestJWKS(t)
	externalJWKS := newTestJWKS(t)
	jwksServer := startJWKSServer(t, externalJWKS)

	trustedIssuers := []TrustedIssuer{{
		IssuerURL:        testExternalIssuer,
		ExpectedAudience: testExternalAudience,
		JWKSURL:          jwksServer.URL + "/jwks",
		AllowedActors:    []string{"ext-agent"},
	}}
	validator := newMultiValidator(t, selfJWKS, trustedIssuers)

	tests := []struct {
		name        string
		claims      func() jwt.Claims
		wantErr     bool
		errContains string
	}{
		{
			name: "nbf ~30s in the future is within leeway",
			claims: func() jwt.Claims {
				c := externalClaims()
				c.NotBefore = jwt.NewNumericDate(time.Now().Add(30 * time.Second))
				return c
			},
		},
		{
			name: "nbf ~5m in the future exceeds leeway",
			claims: func() jwt.Claims {
				c := externalClaims()
				c.NotBefore = jwt.NewNumericDate(time.Now().Add(5 * time.Minute))
				return c
			},
			wantErr:     true,
			errContains: "claims validation failed",
		},
		{
			// go-jose's leeway also widens exp, so a naive implementation would
			// accept this. validateExternalToken's independent strict check
			// must reject it instead.
			name: "expired ~30s ago is rejected by the strict expiry check, not the leeway",
			claims: func() jwt.Claims {
				c := externalClaims()
				c.Expiry = jwt.NewNumericDate(time.Now().Add(-30 * time.Second))
				return c
			},
			wantErr:     true,
			errContains: "subject token has expired",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			rawToken := externalJWKS.signToken(t, tt.claims(), map[string]any{"azp": "ext-agent"})
			result, err := validator.Validate(context.Background(), rawToken)

			if tt.wantErr {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.errContains)
				assert.Nil(t, result)
				return
			}
			require.NoError(t, err)
			require.NotNil(t, result)
		})
	}
}

func TestMultiIssuerTokenValidator_DiscoveryFailures(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		handler     func(w http.ResponseWriter, r *http.Request)
		errContains string // additional substring checked beyond "OIDC discovery failed"
	}{
		{
			name: "non-200",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
			},
		},
		{
			name: "malformed doc",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte("{not-json"))
			},
		},
		{
			name: "issuer mismatch",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(map[string]string{
					"issuer":   "https://different-issuer.example.com",
					"jwks_uri": "http://" + r.Host + "/jwks",
				})
			},
			errContains: "does not match expected issuer",
		},
		{
			name: "missing jwks_uri",
			handler: func(w http.ResponseWriter, r *http.Request) {
				// Issuer must match so discovery reaches the jwks_uri check.
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(map[string]string{
					"issuer": "http://" + r.Host,
				})
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			selfJWKS := newTestJWKS(t)
			externalJWKS := newTestJWKS(t)

			mux := http.NewServeMux()
			mux.HandleFunc("/.well-known/openid-configuration", tt.handler)
			server := httptest.NewServer(mux)
			t.Cleanup(server.Close)

			trustedIssuers := []TrustedIssuer{{
				IssuerURL:        server.URL,
				ExpectedAudience: testExternalAudience,
				// JWKSURL left empty to force discovery.
			}}
			validator := newMultiValidator(t, selfJWKS, trustedIssuers)

			claims := externalClaims()
			claims.Issuer = server.URL
			rawToken := externalJWKS.signToken(t, claims, nil)

			result, err := validator.Validate(context.Background(), rawToken)
			require.Error(t, err)
			assert.Contains(t, err.Error(), "OIDC discovery failed")
			if tt.errContains != "" {
				assert.Contains(t, err.Error(), tt.errContains)
			}
			assert.Nil(t, result)
		})
	}
}

func TestMultiIssuerTokenValidator_KidMismatch(t *testing.T) {
	t.Parallel()

	selfJWKS := newTestJWKS(t)
	externalJWKS := newTestJWKS(t)
	jwksServer := startJWKSServer(t, externalJWKS)

	trustedIssuers := []TrustedIssuer{{
		IssuerURL:        testExternalIssuer,
		ExpectedAudience: testExternalAudience,
		JWKSURL:          jwksServer.URL + "/jwks",
	}}
	validator := newMultiValidator(t, selfJWKS, trustedIssuers)

	// Sign with a key whose public half is NOT in the served JWKS and whose
	// kid does not match any served key. The kid lookup misses, so the
	// validator falls back to trying every served key and fails verification.
	unknownKey := newECDSAJWK(t, "unknown-kid")
	claims := externalClaims()
	rawToken := signWithJWK(t, unknownKey, jose.ES256, claims)

	result, err := validator.Validate(context.Background(), rawToken)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "signature verification failed")
	assert.Nil(t, result)
}

// TestValidateJWKSURL exercises validateJWKSURL directly: this is the SSRF
// guard applied in resolveJWKS to every JWKS URL for a given issuer, whether
// hand-configured on TrustedIssuer or resolved via discovery. The equivalent
// check on redirect hops (networking.SameHostRedirectPolicy) and the
// dial-time IP guard (networking.NewHostScopedClientBuilder) are exercised
// via the networking package's own tests, not here. These cases all pass
// insecureAllowHTTP=false, allowPrivateIPs=false — the strict defaults —
// since every other test in this file goes through newMultiValidator, which
// sets both permissive flags on its test issuers to reach their httptest
// servers over plain HTTP on loopback.
func TestValidateJWKSURL(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		url     string
		wantErr string
	}{
		{name: "https accepted", url: "https://issuer.example.com/jwks"},
		{name: "http rejected", url: "http://issuer.example.com/jwks", wantErr: "must use HTTPS"},
		{name: "loopback IP literal rejected", url: "https://127.0.0.1/jwks", wantErr: "private or loopback"},
		{name: "private IP literal rejected", url: "https://10.1.2.3/jwks", wantErr: "private or loopback"},
		{name: "unparseable URL rejected", url: "://not-a-url", wantErr: "invalid URL"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			err := validateJWKSURL(tt.url, false, false)
			if tt.wantErr == "" {
				assert.NoError(t, err)
				return
			}
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantErr)
		})
	}
}

// TestMultiIssuerTokenValidator_FetchJWKS exercises fetchJWKS's error paths
// directly against an httptest server, bypassing OIDC discovery.
func TestMultiIssuerTokenValidator_FetchJWKS(t *testing.T) {
	t.Parallel()

	// Built once: maxJWKSKeys+1 distinct public keys for the "too many keys" case.
	tooManyKeys := make([]jose.JSONWebKey, maxJWKSKeys+1)
	for i := range tooManyKeys {
		key := newECDSAJWK(t, fmt.Sprintf("k%d", i))
		tooManyKeys[i] = key.Public()
	}

	tests := []struct {
		name    string
		handler http.HandlerFunc
		wantErr string
	}{
		{
			name: "non-200 response",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
			},
			wantErr: "returned status 500",
		},
		{
			name: "malformed JSON body",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte("{not-json"))
			},
			wantErr: "failed to parse JWKS",
		},
		{
			name: "zero keys",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(jose.JSONWebKeySet{Keys: []jose.JSONWebKey{}})
			},
			wantErr: "contains no keys",
		},
		{
			name: "too many keys",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(jose.JSONWebKeySet{Keys: tooManyKeys})
			},
			wantErr: "too many keys",
		},
	}

	selfJWKS := newTestJWKS(t)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			mux := http.NewServeMux()
			mux.HandleFunc("/jwks", tt.handler)
			srv := httptest.NewServer(mux)
			t.Cleanup(srv.Close)

			trustedIssuers := []TrustedIssuer{{
				IssuerURL:        testExternalIssuer,
				ExpectedAudience: testExternalAudience,
				JWKSURL:          srv.URL + "/jwks",
			}}
			validator := newMultiValidator(t, selfJWKS, trustedIssuers)

			_, err := validator.fetchJWKS(context.Background(), validator.issuers[testExternalIssuer])
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantErr)
		})
	}
}

// TestMultiIssuerTokenValidator_JWKSCacheExpiryTriggersRediscovery confirms
// that once the cached JWKS expires, resolveJWKS re-runs OIDC discovery
// rather than reusing the stale jwksURL — required because an issuer may
// rotate its JWKS endpoint. The cache is forced to look expired by mutating
// jwksExp directly rather than waiting out jwksCacheTTL.
func TestMultiIssuerTokenValidator_JWKSCacheExpiryTriggersRediscovery(t *testing.T) {
	t.Parallel()

	selfJWKS := newTestJWKS(t)
	externalJWKS := newTestJWKS(t)

	var discoveryCount, jwksCount atomic.Int32
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, r *http.Request) {
		discoveryCount.Add(1)
		base := "http://" + r.Host
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"issuer": base, "jwks_uri": base + "/jwks"})
	})
	mux.HandleFunc("/jwks", func(w http.ResponseWriter, _ *http.Request) {
		jwksCount.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(externalJWKS.publicJWKS())
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	trustedIssuers := []TrustedIssuer{{
		IssuerURL:        srv.URL,
		ExpectedAudience: testExternalAudience,
		AllowedActors:    []string{"ext-agent"},
		// JWKSURL left empty to force discovery.
	}}
	validator := newMultiValidator(t, selfJWKS, trustedIssuers)

	tokenWithID := func(jti string) string {
		claims := externalClaims()
		claims.Issuer = srv.URL
		claims.ID = jti
		return externalJWKS.signToken(t, claims, map[string]any{"azp": "ext-agent"})
	}

	_, err := validator.Validate(context.Background(), tokenWithID("jti-1"))
	require.NoError(t, err)
	assert.Equal(t, int32(1), discoveryCount.Load())
	assert.Equal(t, int32(1), jwksCount.Load())

	// Force the cache to look expired without waiting jwksCacheTTL out.
	issuerConfig := validator.issuers[srv.URL]
	issuerConfig.mu.Lock()
	issuerConfig.jwksExp = time.Now().Add(-time.Second)
	issuerConfig.mu.Unlock()

	_, err = validator.Validate(context.Background(), tokenWithID("jti-2"))
	require.NoError(t, err)
	assert.Equal(t, int32(2), discoveryCount.Load(),
		"expired cache with no configured JWKSURL must trigger re-discovery")
	assert.Equal(t, int32(2), jwksCount.Load())
}
