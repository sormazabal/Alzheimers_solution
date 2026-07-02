You are an expert technical analyst and technical writer. Your task is to analyse the codebase, documentation, and configuration files in the current repository to generate a comprehensive project summary and presentation outline.

Please output a markdown document that strictly follows the structure below. Do not use the content from the structural examples; they are just to show you the expected format. Extract all actual facts, metrics, architectures, and client goals directly from the repository you are scanning. 

If any section cannot be populated because the data does not exist in the repository, explicitly state "Information not available in the current repository" for that section rather than inventing details.

Required structure and guidelines:

# [Project name]

**Subtitle:** [Generate a one-sentence summary of what the project does]
**Presenter:** [Extract author or maintainer name, if available]

## Slide 2: Outline
- Project overview and client value
- Client pains and system challenges
- Proposed solution
- Dataset overview
- Architecture details
- Technical implementation
- Cost
- Conclusion

## Slide 3: Project overview and client value
**Overview:** [Summarise the core purpose and function of the project]
**Key objective:** [Identify the primary goal or problem it solves]
**Client value:** [Explain why this solution is critical for the client. Detail the strategic, financial, or operational impact it delivers]
**Status:** [Assess current development state, e.g., MVP, production, deprecated]
**Category:** [Identify the broad domain, e.g., data science, infrastructure, clinical research]

## Slide 4: Client pains and system challenges
Provide a breakdown of the specific difficulties the client faces and the technical limitations causing them.
**Client pain points:**
- [Bullet points detailing the specific business, operational, or financial pains the client experiences, such as budget overruns, compliance risks, or resource drains]
**Systemic challenges:**
- [Bullet points explaining the technical, manual, or environmental hurdles that cause the client pains, such as manual data gathering or a lack of systematic scoring]

## Slide 5: Proposed solution
- [Bullet points explaining how the project directly addresses the identified client pains and systemic challenges]
- [Highlight key features and automation steps]

## Slide 6: Architecture details
Provide a layered breakdown of the system.
**Layer 1 - User interface and interaction:** [How users or other systems interact with it]
**Layer 2 - System logic:** [Core modules, packages, and their sequential operations]
**Layer 3 - Data sources:** [External APIs, databases, or local files used]

Include a brief text or JSON example showing typical data inputs and outputs if available in the documentation or test files.

## Slide 7: Dataset overview
Describe any data ingested, processed, or simulated by the system.
Create a markdown table with the columns: Data source, What it contains, and How it is acquired.
List key dataset characteristics, including privacy or security considerations.

## Slide 8: Technical implementation and demo
**Environment:** [List language versions and package managers, e.g., Python >= 3.11, uv, npm]
**Integration:** [List key third-party services, APIs, or LLM providers configured]
**Demo command:** [Provide the CLI command to run or test the project based on the documentation]

## Slide 9: Results and evaluation
Provide any available metrics, benchmark results, or test coverage statistics found in the repository. If the project has distinct evaluation tracks, separate them into subheadings.

## Slide 10: Cost analysis
Provide an analysis of the infrastructure, API, or compute costs required to run the project, if documented. 

## Slide 11: Roadmap and next steps
- [List completed milestones]
- [List future planned features or phase goals found in issues, TODOs, or roadmap files]

## Slide 12: Conclusion and key takeaways
- [Summarise the value proposition and final thoughts based on how the repository solves the initial client pains]