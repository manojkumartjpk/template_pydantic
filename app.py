import torch
from transformers import pipeline
from pydantic import BaseModel, Field
import inferless

@inferless.request
class RequestObjects(BaseModel):
    prompt: str = Field(default="a horse near a beach")

@inferless.response
class ResponseObjects(BaseModel):
    generated_txt: str = Field(default='Test output')

class InferlessPythonModel:
    def initialize(self):
        self.generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M",device=0)
    def infer(self, inputs: RequestObjects):
        pipeline_output = self.generator(inputs.prompt, do_sample=True, min_length=128)
        generateObject = ResponseObjects(generated_txt = pipeline_output[0]["generated_text"])
        return generateObject
    def finalize(self):
        self.generator = None
