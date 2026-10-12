import { describe, expect, it } from "vitest";
import * as api from "../src/index.js";

describe("Pi Host public API", () => {
  it("exports the Pi research harness and transport without legacy Router workflows", () => {
    expect(api).toMatchObject({
      createAuditedModels: expect.any(Function),
      createAuditedProviderController: expect.any(Function),
      createResearchAgentHarness: expect.any(Function),
      loadTimeSigmaProviderConfig: expect.any(Function),
      PythonWorkerClient: expect.any(Function),
      runProviderPreflight: expect.any(Function),
    });
    for (const name of [
      "createProviderDecisionSession",
      "createProviderOnboardingSession",
      "createTimeSigmaDecisionTools",
      "runG5bHybridDriver",
      "runOnboardingDriver",
      "runPairedFixture",
    ]) {
      expect(api).not.toHaveProperty(name);
    }
  });
});
