import { createModelRegistry } from "./registry";
import { deepseekV4 } from "./deepseek-v4";
import { minimaxM3 } from "./minimax-m3";

// Register verified model modules here. No disabled placeholder models.
export const modelRegistry = createModelRegistry([deepseekV4, minimaxM3]);
