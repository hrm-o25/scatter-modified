import { useCallback, useMemo } from "react";
import { Config, Translations, Cluster } from "@/types";

let missing: { [key: string]: boolean } = {};

const JapaneseUI: { [key: string]: string } = {
  Argument: "議論",
  "Original comment": "元のコメント",
  "Representative arguments": "代表的な議論",
  "Open full-screen map": "全画面地図を開く",
  "Back to report": "レポートに戻る",
  "Hide labels": "ラベルを非表示にする",
  "Show labels": "ラベルを表示",
  "Show filters": "フィルターを表示",
  "Hide filters": "フィルターを非表示",
  "Min. votes": "最小投票数",
  Consensus: "コンセンサス",
  Showing: "表示中",
  arguments: "議論",
  "Reset zoom": "ズームをリセット",
  "Click anywhere on the map to close this":
    "このメッセージを閉じるには地図のどこかをクリックしてください",
  "Click on the dot for details": "詳細を見るには点をクリックしてください",
  agree: "同意する",
  disagree: "同意しない",
  Language: "言語",
  English: "英語",
  "of total": "合計",
  Overview: "概要",
  "Cluster analysis": "クラスター分析",
  "Representative comments": "代表的なコメント",
  Introduction: "導入",
  Clusters: "クラスター",
  Appendix: "付録",
  "This report was generated using an AI pipeline that consists of the following steps":
    "このレポートは、以下のステップで構成されるAIパイプラインを使用して生成されました",
  Step: "ステップ",
  extraction: "抽出",
  "show code": "コードを表示",
  "hide code": "コードを非表示",
  "show prompt": "プロンプトを表示",
  "hide prompt": "プロンプトを非表示",
  embedding: "埋め込み",
  clustering: "クラスタリング",
  labelling: "ラベリング",
  takeaways: "まとめ",
  overview: "概要",
};

const useTranslatorAndReplacements = (
  config: Config,
  translations: Translations,
  clusters: Cluster[]
) => {
  const languages = useMemo(() => {
    return [{ name: "日本語", flag: "JP" }];
  }, []);

  const fixLocalLang = useMemo(() => {
    return (x: string) => x;
  }, []);

  const { replaceAll, manualChanges } = useMemo(() => {
    const memory: { [input: string]: string } = {};
    const replacements = config?.visualization?.replacements || [];
    const manualChanges: { from: string; to: string }[] = [];
    let trackChanges = true;
    const replaceAll = (inputString: string) => {
      if (replacements.length === 0) return inputString;
      if (memory[inputString]) return memory[inputString];
      let modifiedString = inputString;
      replacements.forEach((pair) => {
        const { replace, by } = pair;
        modifiedString = modifiedString.split(replace).join(by);
      });
      if (trackChanges && modifiedString !== inputString)
        manualChanges.push({ from: inputString, to: modifiedString });
      memory[inputString] = modifiedString;
      return modifiedString;
    };
    clusters.forEach((cluster) => {
      replaceAll(cluster.cluster);
      replaceAll(cluster.takeaways || "");
      cluster.arguments.forEach((arg) => {
        replaceAll(arg.argument);
      });
    });
    trackChanges = false;
    return { replaceAll, manualChanges };
  }, []);

  const t = useCallback(
    (txt?: string) => {
      if (!txt) return txt;
      const result = JapaneseUI[txt];
      if (!result) {
        console.log(`Missing translation for "${txt}"`);
        return txt;
      }
      return result;
    },
    []
  );

  return { languages, t, manualChanges };
};

export type Translator = ReturnType<typeof useTranslatorAndReplacements>;

export default useTranslatorAndReplacements;
