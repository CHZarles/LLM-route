import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig