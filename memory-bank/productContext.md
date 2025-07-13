# Product Context

This file provides a high-level overview of the project and the expected product that will be created. Initially it is based upon projectBrief.md (if provided) and all other available project-related information in the working directory. This file is intended to be updated as the project evolves, and should be used to inform all other modes of the project's goals and context.
2025-07-13 01:22:09 - Log of updates made will be appended as footnotes to the end of this file.

## Project Goal

Ferrous-LLM is a Rust library providing a unified interface for interacting with multiple Large Language Model providers (OpenAI, Anthropic, Ollama). The project aims to create a modular, type-safe, and performant abstraction layer that allows developers to easily switch between different LLM providers while maintaining consistent APIs.

## Key Features

-   **Multi-Provider Support**: Unified interface for OpenAI, Anthropic, and Ollama providers
-   **Modular Architecture**: Separate crates for core functionality and each provider
-   **Type Safety**: Leverages Rust's type system for safe LLM interactions
-   **Streaming Support**: Real-time streaming capabilities for chat completions
-   **Memory Management**: Dedicated memory crate for conversation context handling
-   **Examples**: Comprehensive examples demonstrating usage patterns

## Overall Architecture

-   **ferrous-llm-core**: Core traits, types, and error handling
-   **ferrous-llm-openai**: OpenAI provider implementation
-   **ferrous-llm-anthropic**: Anthropic provider implementation
-   **ferrous-llm-ollama**: Ollama provider implementation
-   **ferrous-llm-memory**: Memory and context management utilities
-   **examples/**: Usage examples and demonstrations
