package anthropic_test

import (
	"testing"

	"maragu.dev/env"
	"maragu.dev/is"

	anthropic "maragu.dev/gai-anthropic"
)

func TestNewClient(t *testing.T) {
	t.Run("can create a new client with a key", func(t *testing.T) {
		client := anthropic.NewClient(anthropic.NewClientOptions{Key: "123"})
		is.NotNil(t, client)
	})
}

func newClient() *anthropic.Client {
	_ = env.Load(".env.test.local")

	return anthropic.NewClient(anthropic.NewClientOptions{Key: env.GetStringOrDefault("ANTHROPIC_KEY", "")})
}
