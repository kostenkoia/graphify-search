import { useState } from "react";

export function useThing(initial: number) {
  const [value, setValue] = useState(initial);
  return { value, setValue };
}
