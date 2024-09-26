import type OpenAI from 'openai'

import { AIResult } from "@isdk/ai-tool";

export function openaiToAIResultChunk(chunk: OpenAI.Chat.Completions.ChatCompletionChunk, params?: any): AIResult<string, OpenAI.Chat.Completions.ChatCompletionChunk> {
  if (chunk instanceof Uint8Array) {
    const s = new TextDecoder().decode(chunk)
    chunk = JSON.parse(s)
  }

  if (chunk.choices.length === 0) return {content: '', options: chunk, finishReason: 'error'}
  const firstChoice = chunk.choices[0]
  const generation_settings = {...params, ...chunk}
  delete generation_settings.choices

  const result: AIResult<string, OpenAI.Chat.Completions.ChatCompletionChunk> = {
    content: firstChoice.delta.content || '',
    options: {generation_settings} as any,
    finishReason: firstChoice.finish_reason
  }
  if (firstChoice.finish_reason) {
    result.stop = true
  }

  return result
}

export function openaiToAIResult(res: OpenAI.Chat.Completions.ChatCompletion, params?: any): AIResult<string, OpenAI.Chat.Completions.ChatCompletion> {
  if (res.choices.length === 0) return {content: '', options: res, finishReason: 'error'}
  const firstChoice = res.choices[0]
  const generation_settings = {...params, ...res}
  delete generation_settings.choices

  const result: AIResult<string, OpenAI.Chat.Completions.ChatCompletion> = {
    content: firstChoice.message.content || '',
    role: firstChoice.message.role,
    options: {generation_settings} as any,
    finishReason: firstChoice.finish_reason
  }

  return result
}
