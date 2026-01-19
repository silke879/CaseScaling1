import os, time, statistics, json
import pandas as pd
import csv
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from index import clean_row_llm, judge_record_llm

# === Laad LLMs voor judging ===
def judgeTest():
    llm1 = ChatOpenAI(
        model="gpt-3.5-turbo-0125",
        api_key=os.getenv("OPENAI_API_KEY"),
        response_format={"type": "json_object"}
    )

    llm2 = ChatOpenAI(
        model="gpt-4o-mini-2024-07-18",
        api_key=os.getenv("OPENAI_API_KEY"),
        response_format={"type": "json_object"},
    )

    llm3 = ChatOpenAI(
        model="gpt-4o-2024-08-06",
        api_key=os.getenv("OPENAI_API_KEY"),
        response_format={"type": "json_object"},
    )

    judges_llms = [
        ("gpt-3.5-turbo", llm1),
        ("gpt-4o-mini", llm2),
        ("gpt-4o", llm3),
    ]


    with open("data/TestDataSMALL.csv", encoding="utf-8-sig") as csvfile:
        reader = list(csv.reader(csvfile, delimiter=','))

        for row in reader:
            start = time.perf_counter()
            result = judge_record_llm(clean_row_llm(row))
            end = time.perf_counter()
            timeElapsed = end - start
            print(result)
            print(llm1.model_name, timeElapsed)


            start = time.perf_counter()
            result = clean_row_llm(row,llm2)
            end = time.perf_counter()
            timeElapsed = end - start
            print(result)
            print(llm2.model_name, timeElapsed)

            start = time.perf_counter()
            result = clean_row_llm(row,llm3)
            end = time.perf_counter()
            timeElapsed = end - start
            print(result)
            print(llm3.model_name, timeElapsed)

        results = []

        for i, row in enumerate(reader):
            if i == 0:
               rows = reader[1:] # skip header

            for name, llm in [
               ("gpt-3.5-turbo", llm1),
                ("gpt-4o-mini", llm2),
                ("gpt-4o", llm3),
            ]:
                start = time.perf_counter()
                _ = clean_row_llm(row, llm)
                duration = time.perf_counter() - start

                results.append({
                "model": name,
                "row": i,
                "duration_sec": duration
            })

        for model in {"gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"}:
            times = [r["duration_sec"] for r in results if r["model"] == model]
            print(model)
            print("Mean:", round(statistics.mean(times), 3))
            print("Median:", round(statistics.median(times), 3))
            print("p95:", round(sorted(times)[int(len(times) * 0.95)], 3))
            print("-" * 30)

        # Zet je results om naar een DataFrame
        df = pd.DataFrame(results)

        # Boxplot van tijd per model
        plt.figure(figsize=(8, 5))
        df.boxplot(column="duration_sec", by="model")
        plt.title("LLM Response Time per Model")
        plt.suptitle("")
        plt.ylabel("Time (seconds)")
        plt.show()




    df = pd.read_csv("data/TestDataSMALL.csv")

    results = []

    for i, row in df.iterrows():
    # 1️⃣ Cleaning met je originele model
        cleaned = clean_row_llm(row.to_csv(index=False))  # gebruikt je standaard llm uit index.py

    # 2️⃣ Loop over alle judges
        for name, judge_llm in judges_llms:
            start = time.perf_counter()
            try:
                judgement = judge_record_llm(cleaned, _llm=judge_llm)
            except Exception:
                judgement = {
                    "time_score": None,
                    "csat_score": None,
                    "overall_score": None
                }
            duration = time.perf_counter() - start

            results.append({
                "row": i,
                "judge_model": name,
                "duration_sec": duration,
                "time_score": judgement.get("time_score"),
                "csat_score": judgement.get("csat_score"),
                "overall_score": judgement.get("overall_score")
            })
            print(f"Row {i} | Judge {name} | Duration: {duration:.2f}s | Overall: {judgement.get('overall_score')}")

# === Analyse ===
    df_results = pd.DataFrame(results)

# Boxplot: latency per judge
    plt.figure(figsize=(8,5))
    df_results.boxplot(column="duration_sec", by="judge_model")
    plt.title("Judging Latency per Model")
    plt.suptitle("")
    plt.ylabel("Time (seconds)")
    plt.show()

# Boxplot: overall_score per judge
    plt.figure(figsize=(8,5))
    df_results.boxplot(column="overall_score", by="judge_model")
    plt.title("Overall Score per Model")
    plt.suptitle("")
    plt.ylabel("Overall Score")
    plt.show()


if __name__ == "__main__":
    judgeTest()