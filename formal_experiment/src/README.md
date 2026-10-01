# Source code

Add only reviewed code needed by the formal protocol. When adapting an exploratory script, record the archived source path and SHA-256 in the run's provenance record; do not overwrite the archived original.

`deepseek_flash.py` and `gpt6_relay.py` are the shared workspace-level LLM
adapters. Both read credentials from `../../.env.llm` when called through
their `from_env_file` constructors.
