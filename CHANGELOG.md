#  (2026-04-25)


### Bug Fixes

* [no ci] old test branch pruned ([b9d6abe](https://github.com/nguyenv217/callai-agentic_core/commit/b9d6abe797aa7022c927da7e05d07be07da07013))
* better configuratoin error handling. can still import placeholders for convenience ([466ea40](https://github.com/nguyenv217/callai-agentic_core/commit/466ea406b5bbb1e76f2f918a5b04f6fb3bb0223b))
* cleaner seperation between runtime loaded tools and user configured ([272a88a](https://github.com/nguyenv217/callai-agentic_core/commit/272a88a9134e38175ab024a9a0872dcc2bba73f8))
* clearer import pattern ([06974f9](https://github.com/nguyenv217/callai-agentic_core/commit/06974f9d20c3fb2e8d6f2d807e96bae516334e32))
* clearer import pattern. supply the subpackages ([619990f](https://github.com/nguyenv217/callai-agentic_core/commit/619990f341acf076daa2fea804c8ab1e1725569f))
* **engine:** async iterator support ([68735db](https://github.com/nguyenv217/callai-agentic_core/commit/68735db5658f48b36c8a1f233d93d2f291090c03))
* **engine:** config.system_prompt should be optional so end-user can quickly build agent with preconfigured system prompt ([8ba1732](https://github.com/nguyenv217/callai-agentic_core/commit/8ba1732bee7680aca643927f8b7cbd8d41078641))
* **engine:** indentation bug ([d4c733c](https://github.com/nguyenv217/callai-agentic_core/commit/d4c733c4865b779bdd11dd206308fb9f4c211e9a))
* **engine:** new concurrent strat ([63652df](https://github.com/nguyenv217/callai-agentic_core/commit/63652dfcdc12468ed6df947fba6c306075231861))
* make `abandon` decision halt loop instead of blindly continue ([250f84a](https://github.com/nguyenv217/callai-agentic_core/commit/250f84aeabc30f175cac7ea74d1b642bf2898e7b))
* **mcp_init:** Fix bugs with MCP config logic to be more strict. ([5651ceb](https://github.com/nguyenv217/callai-agentic_core/commit/5651cebc46b85ea8da90b54fdb41593f3a0f726b))
* **mcp_init:** remove 'toolset' modifying behaviour. This should be exclusively user-configured ([5b8989c](https://github.com/nguyenv217/callai-agentic_core/commit/5b8989c920b5631d2503bfb16deaaa33660bde71))
* **mcp:** More precise MCP configuration requirement. Will raise error and log if invalid combinations ([bcc368f](https://github.com/nguyenv217/callai-agentic_core/commit/bcc368fb163feef9d042fa6d614b4cc11937786b))
* **memory:** deprecate truncation by popping old messages to preserve context ([1487350](https://github.com/nguyenv217/callai-agentic_core/commit/1487350eedd5262af9268cc73c40774ee3b4e2ae))
* minor redundancy ([a839c6e](https://github.com/nguyenv217/callai-agentic_core/commit/a839c6e923cdde65e98955eb37f04d2c1495e0dc))
* **naming:** fix tests failed because of naming inconsistentcies ([16d85df](https://github.com/nguyenv217/callai-agentic_core/commit/16d85df951d344c68d4406c84b59836aaadd8323))
* **openai:** async client by default ([2a76691](https://github.com/nguyenv217/callai-agentic_core/commit/2a76691880e858c6f2756feac4f16def03c07667))
* **openai:** hotfix to compatible with older sync clients ([10f5193](https://github.com/nguyenv217/callai-agentic_core/commit/10f5193ad94db4f8e0f3f558afa97d7f7468f176))
* **RAG:** `__init__` importing too much will raise importerror if not installed extras. ([f9e7411](https://github.com/nguyenv217/callai-agentic_core/commit/f9e7411dfa4485ef6cb794c41031fbe65cd15abb))
* **RAG:** default async openai ([7050139](https://github.com/nguyenv217/callai-agentic_core/commit/7050139f8cd48e8052bfa3255635aac909460ee6))
* redunant error check logic ([4c99551](https://github.com/nguyenv217/callai-agentic_core/commit/4c9955186424900f169e2fbd8fa8e88449f412d2))
* redundant logic and bug in engine for nonduplicated tools ([64f148f](https://github.com/nguyenv217/callai-agentic_core/commit/64f148fda57938fe04ff52cbbeda3650eb117770))
* redundant manager vars ([ece4063](https://github.com/nguyenv217/callai-agentic_core/commit/ece406343d9b5578bc69dda8b85ce1b4da2eb415))
* rename 'tool_manager' to 'tools' ([3b2891c](https://github.com/nguyenv217/callai-agentic_core/commit/3b2891ca8c63094c1fdb39ef34495bcd8bfad777))
* restructure discovery tools and remove modifying toolset directly. rename get_tools -> get_tools_from_toolset to promote non-modifying of user-configuration ([830e80e](https://github.com/nguyenv217/callai-agentic_core/commit/830e80ed813a783921c453c53816e1e3a1ad8dc5))
* **test:** add missing conf test file ([2aff4b7](https://github.com/nguyenv217/callai-agentic_core/commit/2aff4b7dbad30cc783d3862f93a5e5764cd750ef))
* **test:** fix test suite parallel tools test ([160efbe](https://github.com/nguyenv217/callai-agentic_core/commit/160efbe493b9d3eb3cfdba51f11baf97ec167987))
* **test:** fix test to abide to new naming ([b519def](https://github.com/nguyenv217/callai-agentic_core/commit/b519def5b840f67c1175b3f41d2f4bdda0f6869f))
* **test:** minor changes to follow new manager interfaces ([f100950](https://github.com/nguyenv217/callai-agentic_core/commit/f100950cd4e1b2d558f3c2223f73c544bb221194))
* to async tests ([37455ba](https://github.com/nguyenv217/callai-agentic_core/commit/37455bafb132ad412cddbab5df6732773889bcb7))
* **tool:** More granular control over tool-loading and no longer anyone remote modifying user-supplied toolsets ([7490d87](https://github.com/nguyenv217/callai-agentic_core/commit/7490d8771987613db7fe714c5be6bb7f107d0f9d))
* wrong versions. not needed yet anyway ([5852850](https://github.com/nguyenv217/callai-agentic_core/commit/5852850ef94c4d8a929d870649f3dcfdbb02c01c))


### Features

* **engine:** Much more granular control over tooling turn and iteration ([f1ce63a](https://github.com/nguyenv217/callai-agentic_core/commit/f1ce63acca00da9d53972c41af173bf1dc34164e))
* **engine:** Much more granular control over tooling turn and iteration ([9bdd887](https://github.com/nguyenv217/callai-agentic_core/commit/9bdd887a6b387d565bf5cd024e317182cfdfe3bf))
* **engine:** new DAGpowered engine ([585f947](https://github.com/nguyenv217/callai-agentic_core/commit/585f94721adb7eb2b6783d30f518c2182bba129b))
* heuristically find reasoning field ([ee5c882](https://github.com/nguyenv217/callai-agentic_core/commit/ee5c882edf86d165fcf090a91f63486faeea051d))
* **interface:** new reasoning field ([372b396](https://github.com/nguyenv217/callai-agentic_core/commit/372b39614c23d6fc229be7bea5240674e5988adc))
* **mcp&tool:** add `extra_env` and `extra_context` for runner config anticipate multiple agent runner with different configs ([ca43773](https://github.com/nguyenv217/callai-agentic_core/commit/ca437730ab5ec96e6fb80df6594f7c3f62e5b5f0))
* **mcp:** Add custom ([34f2058](https://github.com/nguyenv217/callai-agentic_core/commit/34f2058953542aa4688db855be2c232f3fd63256))
* **mcp:** allow custom hook be registered with toolmanager for MCP server death ([e9a5d3d](https://github.com/nguyenv217/callai-agentic_core/commit/e9a5d3deb0d3617665c69a1e40bea342eb79aba8))
* **mcp:** move MCP to tools/mcp/ for clearer seperation + singleton for session spawning to avoid dup ([aee5401](https://github.com/nguyenv217/callai-agentic_core/commit/aee5401cce8087b6c37caa1e87ecc426d690dfc5))
* **memory:** explicit strategy for trunctation memory. user can supply this for their own domain-specific truncation strat ([7e0038a](https://github.com/nguyenv217/callai-agentic_core/commit/7e0038a09c1e5ff47e919f231b33b51ba936f9de))
* no truncation memory strat to preserve prompt caching ([351e5c1](https://github.com/nguyenv217/callai-agentic_core/commit/351e5c1dfe126d80922ab38eab3f52ac22147637))
* **openai:** more versatile client initialization ([717b5b2](https://github.com/nguyenv217/callai-agentic_core/commit/717b5b2061b857c421c80b7fbc8b0e176ea61654))
* **RAG:** concrete implementation of common databases ([f42108b](https://github.com/nguyenv217/callai-agentic_core/commit/f42108b8351a40ea54185dc336316eb63a2c564f))
* **RAG:** simple rag tooling package ([41bd09a](https://github.com/nguyenv217/callai-agentic_core/commit/41bd09a5516f6ee888ed41efdf2269297f7ee363))
* **reasoning:** done new feature reasoning with option for observer to see it at start of tool execution ([1b28040](https://github.com/nguyenv217/callai-agentic_core/commit/1b28040498b0f359d152faf59d6e1e577d78d912))
* safer config ([ff4db2a](https://github.com/nguyenv217/callai-agentic_core/commit/ff4db2a9874cd053e9b77b7b7702a47b4fabee77))
* **tool:** allow add toolset after init ([92332ea](https://github.com/nguyenv217/callai-agentic_core/commit/92332eabe3853aca99473898680ed1f8ab6576d5))
* **tool:** allows extra context to be injected to toolmanager.execute() ([6d7dfe1](https://github.com/nguyenv217/callai-agentic_core/commit/6d7dfe187fe5f33b1afb1e1093fe688381994912))
* **tool:** more control for tooling, now support decision feedback before executing tools directly through AgentEventObserver ([ca2269b](https://github.com/nguyenv217/callai-agentic_core/commit/ca2269bc1c4b9748dd328eed196398a171ead848))
* **tool:** supports for a dynamically injected system prompt along toolset. this is aanticipated for auto routing tasks ([a4d2469](https://github.com/nguyenv217/callai-agentic_core/commit/a4d246956e03e024d58c736310ebe28d4f17b544))
* **tool:** user configurable `max_chars` for tool results ([7bab473](https://github.com/nguyenv217/callai-agentic_core/commit/7bab47351465748d4e83439f7840f20e5aa05768))

# (2026-04-26)

### Breaking Changes
* **API Interface**: `chat()` and `AgentRunner.run_turn()` now return a structured `AgentResponse` object instead of a string or dict. Use `.text` to access the response content.

### Features
* **Session Management**: Added `session_id` to `chat()` to enable persistence of Memory and MCP connections across turns.
* **Lifecycle Management**: `AgentRunner` now supports the async context manager protocol (`async with`), ensuring all MCP subprocesses are cleaned up immediately.
* **Performance**: `GlobalMCPRegistry` now uses granular per-server locks, preventing global bottlenecks during concurrent server initialization in web environments.

### Bug Fixes

* for transparency no longer try to fix double serialization. this is better controlled through explicit prompting and provider-side ([d33dd15](https://github.com/nguyenv217/callai-agentic_core/commit/d33dd1547c370ff26b4e85631a95b02d17068592))

---

# [](https://github.com/nguyenv217/callai-agentic_core/compare/v0.5.0...v) (2026-04-29)


### Bug Fixes

* await close mcp ([0a592c7](https://github.com/nguyenv217/callai-agentic_core/commit/0a592c715660058a4b1f0ef90ed1dde4e0547125))
* bug where nodes could still run when terminal states are reached due to delay mechanism ([679b2df](https://github.com/nguyenv217/callai-agentic_core/commit/679b2df857a06704ae5f24171fd7e1c7b32a848d))
* circuit break the failed init servers ([a3a5a57](https://github.com/nguyenv217/callai-agentic_core/commit/a3a5a57197841c20ee7ad0cbd579531348e3fada))
* clean up example ([de169a1](https://github.com/nguyenv217/callai-agentic_core/commit/de169a1eecd400693ffe4179c7d4899591780001))
* corrected mistake. mcpclient alreayd register itself with release. will just document usages instead ([9228b1b](https://github.com/nguyenv217/callai-agentic_core/commit/9228b1b72668e3441ab9130f7d90fba87bdd474e))
* for transparency no longer try to fix double serialization. this is better controlled through explicit prompting and provider-side ([d33dd15](https://github.com/nguyenv217/callai-agentic_core/commit/d33dd1547c370ff26b4e85631a95b02d17068592))
* mcp shutdown concurrently and await cancellation task so the they dont race with the OS in unexpected manners ([10c01a9](https://github.com/nguyenv217/callai-agentic_core/commit/10c01a93b8f35fe9dffe1ebdbb1cce2825fd9abd))
* **mcp:** automatic shutoff for registry ([69c22cf](https://github.com/nguyenv217/callai-agentic_core/commit/69c22cfd625650c36e599bdfe52e210bbae3830c))
* **mcp:** enhance mcp env parsing and premature spawn retry and concurrent retry when 2 agentrunner thread dispatch at same time and identity locks are not held ([e6ea69d](https://github.com/nguyenv217/callai-agentic_core/commit/e6ea69db5bc2d073ddef647f22d675a8a2564040))
* **mcp:** remove hacky stdio routing to builtin errlog ([b4d542c](https://github.com/nguyenv217/callai-agentic_core/commit/b4d542c6517849be6bf6ccb22cccfa801a12c6fc))
* multiple mcp client for 1 toolmanager ([3fee024](https://github.com/nguyenv217/callai-agentic_core/commit/3fee02485e98ad781434b2b8c2550042861419c8))
* registry cleanup ([2feb615](https://github.com/nguyenv217/callai-agentic_core/commit/2feb61569ff574d557685fc65101eef6e39ac1d2))


### Features

* command can inherit envs as well ([abd9d67](https://github.com/nguyenv217/callai-agentic_core/commit/abd9d6701e86f90b4b44dc918807dec577a60c94))
* **mcp:** allow routing stdio to avoid conflicts ([deaa1da](https://github.com/nguyenv217/callai-agentic_core/commit/deaa1da497bb578be3200b5fbeaa6e25da50c230))
* **streaming:** iter 1 streaming with openai, new `stream_engine` ([668e0a4](https://github.com/nguyenv217/callai-agentic_core/commit/668e0a4950cd10aae6f8b6a213ae8b9acabecd32))
* **streaming:** iter 2 better error handling ([adadf38](https://github.com/nguyenv217/callai-agentic_core/commit/adadf38605eb5a0f127cc76f2522d4af5210999e))
# [](https://github.com/nguyenv217/callai-agentic_core/compare/v0.6.0...v) (2026-05-01)


### Bug Fixes

* also remove standby registry when server is disconnected ([5dd5679](https://github.com/nguyenv217/callai-agentic_core/commit/5dd567983cd8742a743acfe35a5eb27fd8f8413e))
* bug multilating dictionary while iterating ([b2c8947](https://github.com/nguyenv217/callai-agentic_core/commit/b2c8947835bb44c9a189aad185f4761a883fc747))
* subagent tool validation before passing to engine ([62a5889](https://github.com/nguyenv217/callai-agentic_core/commit/62a5889fa20bc77eb286104f0a424620a20bb43c))
* typos in tests ([c7208de](https://github.com/nguyenv217/callai-agentic_core/commit/c7208de0bd50f626b05d1543a25dc252346af0dd))


### Features

* `DAGeventObserver` record the original dagresponse object instead of dict ([25a3806](https://github.com/nguyenv217/callai-agentic_core/commit/25a3806f02b9bf50ab2f057f3d7110a4a13f06d2))
* checkpoint/resume failed graphs ([ef8c8c9](https://github.com/nguyenv217/callai-agentic_core/commit/ef8c8c9472b05ca4c6e8cc07010e6e70de68b941))
* new ops on last iteration - extend. and will raise `LastIterationError` to better accomodate deeply nested swarm workflow ([aabb745](https://github.com/nguyenv217/callai-agentic_core/commit/aabb745e425cc745d8ab540be8271d28f138d1fd))
* validate tools config for spawn sbuagent tool as well ([c528e7d](https://github.com/nguyenv217/callai-agentic_core/commit/c528e7d21a3c18ad13e5c5a77ab659ac2fb9c6f0))
