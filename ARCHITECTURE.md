# Architecture & Design Document

## System Overview

The LLM Red Teaming Framework is designed with modularity, extensibility, and systematic coverage in mind. It follows a pipeline architecture where each component has a clear responsibility.

## Core Components

### 1. Prompt Generator (`PromptGenerator`)

**Purpose**: Systematically generate adversarial test prompts across threat categories.

**Key Features**:
- Template-based generation for consistency
- Parameterized prompts for variations
- Category-specific strategies
- Metadata tracking for each test case

**Design Decisions**:
- Templates are organized by threat category and subcategory
- Substitution variables allow for diverse test cases from single templates
- Extensible design allows adding new categories easily

### 2. Attack Strategies Module

**Purpose**: Implement sophisticated adversarial techniques to test model robustness.

**Components**:

#### PromptMutator
- Character substitution (leet speak, homoglyphs)
- Encoding obfuscation (base64, hex, ROT13)
- Language mixing
- Typo injection
- Formatting tricks

#### JailbreakTechniques
- Role-playing attacks
- Hypothetical scenarios
- Authority appeals
- Prefix injection
- Refusal suppression

#### MultiTurnAttack
- Gradual escalation
- Trust building
- Context manipulation
- Appeal to authority

#### AdaptiveAttacker
- Learns from model responses
- Adjusts strategy based on refusal patterns
- Maintains attack history
- Suggests next attack vector

**Design Rationale**:
- Separation of concerns: each technique is isolated
- Composability: techniques can be combined
- Observability: all mutations are tracked

### 3. LLM Adapters (`llm_adapters.py`)

**Purpose**: Provide unified interface to various LLM providers.

**Architecture**:
```
         LLMInterface (Abstract Base Class)
                    |
      ┌─────────────┼──────────────┐
      |             |              |
OpenAIAdapter  AnthropicAdapter  LocalModelAdapter
      |             |              |
 AzureOpenAI   HuggingFace    CustomAdapter
```

**Design Benefits**:
- **Abstraction**: User code doesn't depend on specific provider
- **Flexibility**: Easy to swap providers
- **Extensibility**: New providers added by implementing interface
- **Testability**: Mock implementations for testing

### 4. Safety Evaluator (`SafetyEvaluator`)

**Purpose**: Assess whether model responses are safe or violate policies.

**Evaluation Methods**:

1. **Rule-Based Evaluation**
   - Fast, deterministic
   - Pattern matching for refusals
   - Response length heuristics
   - Keyword detection

2. **LLM-as-Judge Evaluation**
   - Nuanced understanding
   - Context-aware
   - Can detect subtle violations
   - More expensive

**Design Choices**:
- Hybrid approach: rules + LLM judge
- Configurable: can use either or both
- Severity scoring for prioritization
- Detailed reasoning for transparency

### 5. Red Team Runner (`RedTeamRunner`)

**Purpose**: Orchestrate the complete red teaming process.

**Responsibilities**:
- Test execution with concurrency control
- Rate limiting
- Result aggregation
- Report generation
- Progress tracking

**Flow**:
```
Test Cases → Query Model → Evaluate Response → Aggregate Results → Generate Report
```

**Concurrency Model**:
- Async/await for efficient I/O
- Batched execution
- Configurable concurrency limits
- Rate limiting between batches

## Data Flow

```
┌─────────────────┐
│ User Config     │
└────────┬────────┘
         │
         v
┌─────────────────┐     ┌──────────────┐
│ Prompt          │────▶│ Test Cases   │
│ Generator       │     └──────┬───────┘
└─────────────────┘            │
                               │
         ┌─────────────────────┘
         │
         v
┌─────────────────┐     ┌──────────────┐
│ Attack          │────▶│ Enhanced     │
│ Strategies      │     │ Test Cases   │
└─────────────────┘     └──────┬───────┘
                               │
                               v
                        ┌──────────────┐
                        │ Red Team     │
                        │ Runner       │
                        └──────┬───────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         v                     v                     v
┌─────────────────┐   ┌──────────────┐    ┌──────────────┐
│ Target Model    │   │ Query        │    │ Rate Limiter │
│ (via Adapter)   │◀──│ Executor     │◀───│              │
└────────┬────────┘   └──────────────┘    └──────────────┘
         │
         │ Response
         v
┌─────────────────┐
│ Safety          │
│ Evaluator       │
└────────┬────────┘
         │
         │ Evaluation Result
         v
┌─────────────────┐     ┌──────────────┐
│ Result          │────▶│ Report       │
│ Aggregator      │     │ Generator    │
└─────────────────┘     └──────┬───────┘
                               │
                               v
                        ┌──────────────┐
                        │ Output       │
                        │ (JSON/HTML)  │
                        └──────────────┘
```

## Thread Safety & Concurrency

### Async Design
- All I/O operations are async
- Non-blocking model queries
- Concurrent test execution
- Efficient resource utilization

### Safety Measures
- Rate limiting to prevent API abuse
- Timeout handling
- Retry logic with exponential backoff
- Connection pooling

## Extensibility Points

### 1. Adding New Threat Categories

```python
# In ThreatCategory enum
NEW_CATEGORY = "new_category_name"

# In PromptGenerator._load_templates()
ThreatCategory.NEW_CATEGORY: [
    {
        "subcategory": "sub1",
        "templates": ["template 1", "template 2"]
    }
]
```

### 2. Adding New Attack Strategies

```python
class NewAttackStrategy:
    def generate(self, base_prompt: str) -> str:
        # Implementation
        return modified_prompt
```

### 3. Adding New LLM Providers

```python
class NewProviderAdapter(LLMInterface):
    async def generate(self, prompt: str) -> str:
        # Implementation
        pass
    
    async def chat(self, messages: List[Dict]) -> str:
        # Implementation
        pass
```

### 4. Custom Evaluation Logic

```python
class CustomEvaluator(SafetyEvaluator):
    async def evaluate_response(self, test_case, response):
        # Custom logic
        return TestResult(...)
```

## Configuration Management

### Hierarchical Configuration
1. Default values (hardcoded)
2. Config file (config.py)
3. Environment variables (.env)
4. Runtime parameters

### Configuration Precedence
Runtime > Environment > Config File > Defaults

## Error Handling Strategy

### Levels
1. **Fatal Errors**: Stop execution (e.g., no API key)
2. **Recoverable Errors**: Retry with backoff (e.g., rate limit)
3. **Test Failures**: Log and continue (e.g., model timeout)

### Logging
- Structured logging with levels
- Test execution audit trail
- Performance metrics
- Error traces

## Security Considerations

### API Key Management
- Environment variables only
- Never hardcode keys
- Separate keys for testing/production
- Minimal permissions

### Data Handling
- Sensitive test cases marked
- Optional PII redaction in reports
- Secure temporary storage
- Cleanup of intermediate files

### Rate Limiting
- Respect provider limits
- Configurable delays
- Adaptive throttling
- Fail-safe mechanisms

## Performance Optimization

### Bottlenecks
1. API latency (network I/O)
2. LLM inference time
3. Evaluation compute

### Optimizations
- Concurrent test execution
- Caching (where applicable)
- Batch processing
- Async I/O

### Scalability
- Stateless design
- Horizontal scaling possible
- Distributed execution support
- Result streaming

## Testing Strategy

### Unit Tests
- Individual component testing
- Mock dependencies
- Edge cases
- Error conditions

### Integration Tests
- End-to-end pipeline
- Real API calls (limited)
- Report generation
- Multi-component interaction

### Performance Tests
- Load testing
- Latency measurement
- Resource usage
- Scalability limits

## Future Enhancements

### Planned Features
1. **Advanced Reporting**
   - HTML dashboard
   - Interactive visualizations
   - Trend analysis over time
   - Comparison across models

2. **Enhanced Evaluation**
   - Multi-judge ensemble
   - Confidence scoring
   - Explainability
   - Human-in-the-loop

3. **Attack Evolution**
   - Genetic algorithms
   - Reinforcement learning
   - Model-specific adaptation
   - Automated discovery

4. **Deployment**
   - CI/CD integration
   - Scheduled testing
   - Alert system
   - API service mode

### Research Directions
- Adversarial training feedback loop
- Cross-model vulnerability patterns
- Transferability analysis
- Safety benchmark standardization

## Design Principles

1. **Modularity**: Each component has single responsibility
2. **Extensibility**: Easy to add new features
3. **Transparency**: Clear reasoning and audit trails
4. **Efficiency**: Optimized for performance
5. **Safety**: Ethical use enforced at design level
6. **Testability**: Comprehensive test coverage
7. **Documentation**: Self-documenting code

## Technology Stack

- **Language**: Python 3.8+
- **Async**: asyncio, aiohttp
- **Data**: pandas, numpy
- **Testing**: pytest, pytest-asyncio
- **APIs**: Various LLM provider SDKs
- **Reports**: JSON, Jinja2 templates

---

This architecture balances flexibility, performance, and safety to provide a robust framework for systematic LLM red teaming.
