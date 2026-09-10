import type { ModelDefinition } from "../registry";
import "./layout.css";
import { DeepSeekV4View } from "./view";
import { HelpModal } from "./reference";
import { CODE_URL, WEIGHTS_URL, PAPER_URL, ASCEND_COMMIT } from "./sources";

export const deepseekV4 = {
  id: "deepseek-v4",
  name: "DeepSeek-V4-Flash",
  resources: [
    { label: "CODE", description: `vllm-ascend @ ${ASCEND_COMMIT.slice(0, 7)}`, url: CODE_URL },
    { label: "PAPER", description: "arXiv:2606.19348", url: PAPER_URL },
    { label: "WEIGHTS", description: "Hugging Face · Flash 284B", url: WEIGHTS_URL },
  ],
  facts: [
    { value: "284B", label: "Flash 总参数量" },
    { value: "13B", label: "每 token 激活参数" },
    { value: "1M", label: "最大上下文 token" },
    { value: "CSA/HCA", label: "Hybrid attention" },
  ],
  View: DeepSeekV4View,
  Reference: HelpModal,
} satisfies ModelDefinition;
