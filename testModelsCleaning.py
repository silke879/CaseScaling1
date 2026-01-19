from io import StringIO
import os, time
import csv
from langchain_openai import ChatOpenAI
from index import clean_row_llm
import statistics
import matplotlib.pyplot as plt
import pandas as pd


def cleaningTest():
    llm1 = ChatOpenAI(
        model="gpt-3.5-turbo-0125",
        api_key=os.getenv("OPENAI_API_KEY"),
        response_format = {"type": "json_object"}
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

    with open("data/TestDataSMALL.csv", encoding="utf-8-sig") as csvfile:
        reader = list(csv.reader(csvfile, delimiter=','))

        for row in reader:
            start = time.perf_counter()
            result = clean_row_llm(row,llm1)
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


if __name__ == "__main__":
    cleaningTest()