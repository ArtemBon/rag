from litellm import completion


class LLMClient:
  """Client for interacting with LLM via LiteLLM."""

  def __init__(self, model_name: str, api_key: str, temperature: float = 0.1, max_tokens: int = 3000):
    self.model_name = model_name
    self.api_key = api_key
    self.temperature = temperature
    self.max_tokens = max_tokens
    self.system_prompt = self._create_system_prompt()

  def answer(self, question: str, context: str) -> str:
    user_prompt = self._build_user_prompt(question, context)
    response = completion(
      model=self.model_name,
      messages=[
        {"role": "system", "content": self.system_prompt},
        {"role": "user", "content": user_prompt}
      ],
      api_key=self.api_key,
      temperature=self.temperature,
      max_tokens=self.max_tokens
    )
    return response.choices[0].message.content.strip()

  def _create_system_prompt(self) -> str:
    return """
      You are a knowledgeable expert assistant.

      Instructions:
      1. Answer questions using ONLY information from the context provided below
      2. You can rephrase and organize information from the context into a coherent answer
      3. DO NOT add any factual claims, statistics, technical details, examples, or specific information that is not explicitly stated in the context
      4. You may use natural connecting words and phrases to structure the answer (like "additionally", "however", "furthermore")
      5. Present the information naturally as your direct expert knowledge
      6. Provide detailed answers using ALL relevant information found in the context
      7. For multi-part questions, address each part using information from the context
      8. Respond in the same language as the question

      CRITICAL: NEVER use these phrases or similar ones:
      - "the context"
      - "the sources"
      - "according to"
      - "based on the information provided"
      - "the information describes"
      - "it is mentioned that"
      - "as stated"
      - "the document says"

      Write as if you directly know this information, not as if you're reading it from somewhere.
  """

  def _build_user_prompt(self, question: str, context: str) -> str:
    return f"""
      Context:
      {context}

      Question: {question}

      Answer:
    """