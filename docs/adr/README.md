# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the Spike-SNN Event Vision Kit project.

## What is an ADR?

An Architecture Decision Record (ADR) is a document that captures an important architectural decision made along with its context and consequences.

## ADR Format

We use the template format:

```markdown
# ADR-XXXX: Title

## Status
[Proposed | Accepted | Rejected | Deprecated | Superseded]

## Context
What is the issue that we're seeing that is motivating this decision or change?

## Decision
What is the change that we're proposing and/or doing?

## Consequences
What becomes easier or more difficult to do because of this change?
```

## Index of ADRs

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-0001](adr-0001-spiking-neural-network-framework.md) | Spiking Neural Network Framework Selection | Accepted | 2025-08-02 |
| [ADR-0002](adr-0002-event-camera-abstraction.md) | Event Camera Hardware Abstraction | Accepted | 2025-08-02 |
| [ADR-0003](adr-0003-neuromorphic-hardware-backends.md) | Neuromorphic Hardware Backend Strategy | Accepted | 2025-08-02 |

## Guidelines

1. **Number consecutively**: ADRs are numbered sequentially (ADR-0001, ADR-0002, etc.)
2. **Use present tense**: Write as if the decision is being made now
3. **Keep it concise**: Focus on the decision, not implementation details
4. **Update status**: Mark as Superseded when replaced by newer ADRs
5. **Link related ADRs**: Reference other ADRs when decisions build on each other