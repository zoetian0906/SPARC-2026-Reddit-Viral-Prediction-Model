{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOuVXeqFPj1um6vzvq8ukM/",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/zoetian0906/SPARC-2026-Reddit-Viral-Prediction-Model/blob/main/app/backend/parse.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "bKpl5Q3St8M7"
      },
      "outputs": [],
      "source": [
        "from __future__ import annotations\n",
        "import os\n",
        "from typing import Literal, Optional\n",
        "from pydantic import BaseModel, Field\n",
        "\n",
        "from langchain_google_genai import ChatGoogleGenerativeAI\n",
        "from langchain_core.prompts import ChatPromptTemplate\n",
        "\n",
        "# 1. Define the exact schema we want the LLM to output\n",
        "class QueryData(BaseModel):\n",
        "    category: Optional[Literal[\n",
        "        \"Food & Cooking\", \"Gaming\", \"Skincare & Beauty\", \"Personal Finance\",\n",
        "        \"Career & Work\", \"Fitness & Health\", \"Mental Health\",\n",
        "        \"Relationships & Advice\", \"Tech & Gadgets\", \"Home & Interior\"\n",
        "    ]] = Field(\n",
        "        default=None,\n",
        "        description=\"The closest matching category. Null if the text doesn't fit any.\"\n",
        "    )\n",
        "    mechanism: Optional[Literal[\"question\", \"showcase\", \"statement\"]] = Field(\n",
        "        default=None,\n",
        "        description=\"'question' if asking something, 'showcase' if showing off a project/item, 'statement' if general commentary. Null if indeterminate.\"\n",
        "    )\n",
        "    location_mentioned: Optional[str] = Field(\n",
        "        default=None,\n",
        "        description=\"Any specific city, state, or geographic location mentioned. Null if none.\"\n",
        "    )\n",
        "\n",
        "# 2. Initialize the model and force it to use the Pydantic schema\n",
        "# We use temperature=0 because we want deterministic extraction, not creative writing\n",
        "llm = ChatGoogleGenerativeAI(model=\"gemini-2.5-flash\", temperature=0)\n",
        "structured_llm = llm.with_structured_output(QueryData)\n",
        "\n",
        "# 3. Create the prompt template\n",
        "prompt = ChatPromptTemplate.from_messages([\n",
        "    (\"system\", \"You are an intelligent natural language parser. Extract the requested fields from the user's text based strictly on the provided schema.\"),\n",
        "    (\"user\", \"{text}\")\n",
        "])\n",
        "\n",
        "# 4. Build the extraction chain\n",
        "parse_chain = prompt | structured_llm\n",
        "\n",
        "def parse_query(text: str) -> dict:\n",
        "    \"\"\"Parse free text into recommendation params using an LLM.\n",
        "\n",
        "    Acts as a drop-in replacement for the regex stub.\n",
        "    post_type is always None because plain text carries no media signal.\n",
        "    \"\"\"\n",
        "    text = text or \"\"\n",
        "    stripped = text.strip()\n",
        "\n",
        "    # The guaranteed baseline dictionary shape\n",
        "    result = {\n",
        "        \"category\": None,\n",
        "        \"post_type\": None,\n",
        "        \"mechanism\": None,\n",
        "        \"location_mentioned\": None,\n",
        "        \"raw_text\": text,\n",
        "    }\n",
        "\n",
        "    if not stripped:\n",
        "        return result\n",
        "\n",
        "    try:\n",
        "        # Invoke the LLM chain to extract the structured data\n",
        "        parsed_data = parse_chain.invoke({\"text\": stripped})\n",
        "\n",
        "        # Populate our standard dictionary with the LLM's findings\n",
        "        result[\"category\"] = parsed_data.category\n",
        "        result[\"mechanism\"] = parsed_data.mechanism\n",
        "        result[\"location_mentioned\"] = parsed_data.location_mentioned\n",
        "\n",
        "    except Exception as e:\n",
        "        # In case of API failure or quota limits, fail gracefully\n",
        "        # and return the empty structure rather than crashing the backend.\n",
        "        print(f\"LLM Parsing failed: {e}\")\n",
        "\n",
        "    return result\n",
        "\n",
        "def location_note(location: str) -> str:\n",
        "    \"\"\"Human-readable disclaimer that the model is not geographic.\"\"\"\n",
        "    return (\n",
        "        f\"You mentioned {location}. Our data isn't geographic, so this is \"\n",
        "        \"general guidance, not specific to that area.\"\n",
        "    )"
      ]
    }
  ]
}