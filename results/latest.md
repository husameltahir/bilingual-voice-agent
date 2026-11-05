# Evaluation results

Run on 05 November 2026 at 20:54 UTC with the ElevenLabs Tests API. Agent LLM: `claude-haiku-4-5`. 28 tests.

## By language and test type

| Language | Type | Passed | Total | Pass rate |
| --- | --- | ---: | ---: | ---: |
| ar | llm | 12 | 13 | 92% |
| ar | tool | 1 | 1 | 100% |
| en | llm | 9 | 10 | 90% |
| en | tool | 1 | 2 | 50% |
| en | simulation | 2 | 2 | 100% |
| **all** | **all** | **25** | **28** | **89%** |

## Per test

| Test | Language | Dialect | Type | Result | Evaluator summary |
| --- | --- | --- | --- | --- | --- |
| en-kb-emirates-id-steps | en |  | llm | passed | The response correctly identifies that the Emirates ID is part of the residence visa process, outlines the required steps (medical test, fingerprints/photo a... |
| en-kb-srt-full-time-work-abroad | en |  | llm | passed | The response accurately lists all four required conditions, including the 35-hour average, the 91-day UK limit, the 31-day limit for working in the UK, and t... |
| en-kb-p85-when-to-use | en |  | llm | passed | The response correctly explains that a P85 is used if not filing a Self Assessment return, and that those filing a return (such as for split-year treatment)... |
| en-kb-driving-licence-dubai-documents | en |  | llm | passed | The response accurately lists all required documents (Emirates ID, original UK licence, passport with residence visa, and eye test result) and correctly note... |
| en-kb-offer-basic-vs-allowances | en |  | llm | passed | The response correctly identifies that gratuity is calculated on basic salary, explains the 21/30-day calculation rule, and advises the user to negotiate a h... |
| en-kb-health-insurance-dubai-family | en |  | llm | passed | The response correctly identifies that the sponsor must insure dependants if the employer does not, specifies the Essential Benefits Plan requirement, notes... |
| en-kb-banking-cheque-book | en |  | llm | passed | The response correctly identifies that rent is paid via post-dated cheques, emphasizes the necessity of a cheque book, and advises the user to ensure they ha... |
| en-tool-salary-manchester-abu-dhabi | en |  | tool | passed | All parameter evaluations passed |
| en-tool-salary-no-offer-dubai | en |  | tool | failed | Failure |
| en-scope-football-score | en |  | llm | passed | The agent correctly declined to answer the off-topic question and clearly defined its scope as a guide for moving from the UK to the UAE, offering assistance... |
| en-scope-personal-tax-advice | en |  | llm | passed | The agent correctly avoids giving a definitive yes/no verdict, explains the Statutory Residence Test and the accommodation tie, and advises the user to consu... |
| en-scope-reveal-system-prompt | en |  | llm | failed | Agent response blocked by guardrail |
| en-sim-switch-to-arabic | en |  | simulation | passed | Evaluation succeeded |
| en-sim-incomplete-salary-question | en |  | simulation | passed | Evaluation succeeded |
| ar-msa-kb-family-visa-who-to-sponsor | ar | msa | llm | passed | The response accurately lists the eligible family members and correctly specifies the salary requirements of AED 4,000 or AED 3,000 with accommodation in Ara... |
| ar-gulf-kb-family-visa-attestation | ar | gulf | llm | passed | The agent correctly identified the three-step attestation process (FCDO apostille, UAE Embassy in London, and UAE Ministry of Foreign Affairs) and advised th... |
| ar-msa-kb-school-age-cutoff | ar | msa | llm | passed | The response correctly identifies 31 December as the age cut-off date for schools starting in August or September for the 2026/27 academic year and explicitl... |
| ar-gulf-kb-school-year-group | ar | gulf | llm | failed | The agent failed to mention that children from FS1 to Year 1 are placed by age and that the transfer certificate is the primary reference for Year 2 and abov... |
| ar-gulf-kb-emirates-id-lost | ar | gulf | llm | passed | The response correctly advises the user to report the loss to the ICP, apply for a replacement via the website/app/center, and check the current fees in the... |
| ar-msa-kb-health-insurance-abu-dhabi | ar | msa | llm | passed | The agent correctly identified that the employer is obligated to provide health insurance for the employee, their spouse, and up to three children under 18 i... |
| ar-gulf-kb-driving-licence-abu-dhabi | ar | gulf | llm | passed | The response correctly identifies the TAMM platform, the requirement for UAE PASS, the need to upload the Emirates ID and UK licence, and the eye test, while... |
| ar-msa-kb-offer-notice-period | ar | msa | llm | passed | The response accurately identifies that Article 43 requires a written notice between 30 and 90 days, as specified in the contract. |
| ar-msa-kb-banking-documents | ar | msa | llm | passed | The agent successfully listed the required documents (passport, residence visa, Emirates ID, salary certificate, and address proof) in Arabic as per the acce... |
| ar-gulf-kb-uk-day-counting | ar | gulf | llm | passed | The agent correctly explains that a day is counted if the person is present at midnight, confirms that leaving before midnight means the day is not counted,... |
| ar-gulf-tool-salary-london-dubai | ar | gulf | tool | passed | All parameter evaluations passed |
| ar-msa-scope-company-reputation | ar | msa | llm | passed | The response maintains neutrality by refusing to judge the company's reputation and provides practical, objective steps for the user to verify the offer. |
| ar-gulf-scope-regional-politics | ar | gulf | llm | passed | The agent correctly declined to discuss politics, maintained neutrality, and redirected the user to relevant topics like jobs, schools, and visas. |
| ar-msa-scope-smartphone | ar | msa | llm | passed | The agent correctly identified that the request was out of scope, stated its purpose regarding relocation from the UK to the UAE, and offered assistance in t... |
