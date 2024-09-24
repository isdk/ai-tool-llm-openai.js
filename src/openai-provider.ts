import OpenAI, {ClientOptions as OpenAIClientOptions} from 'openai'
import { AIResult, AsyncFeatures, type CancelableAbility, NotFoundError, TaskAbortController, makeToolFuncCancelable, throwError } from "@isdk/ai-tool";
import { LLMProvider, AIOptions } from "@isdk/ai-tool-llm";
import {
  openaiToAIResultChunk,
  openaiToAIResult,
} from "./options";

export const OpenaiProviderName = 'openai'
const OpenaiJsonSchemaModels = [
  'gpt-4o-2024-08-06',
  'gpt-4o-mini-2024-07-18',
  'gpt-4o-mini',
  'chatgpt-4o-latest',
  'o1-preview',
  'o1-preview-2024-09-12',
  'o1-mini',
]

export interface OpenaiProvider extends CancelableAbility {}
export class OpenaiProvider extends LLMProvider {
  rule = [
    /^gpt-4o(-2024-(05-13|08-06)|-mini(-2024-07-18)?)?$/,
    /^chatgpt-4o-latest$/,
    /^o1-preview(-2024-09-12)?$/,
    /^o1-mini(-2024-09-12)?$/,
    /^gpt-(4|3\.5)(-[\w\d]+)*?$/,
    /^text-embedding-3-(large|small)/,
    'text-embedding-ada-002',
  ]

  async processModelOptions(model: string, prompt: any, options: AIOptions) {
    const vIsString = typeof prompt === 'string'
    const baseURL = options.apiUrl || this.apiUrl
    // const apiKey = options.apiKey || this.apiKey
    if (!model) {
      throw new NotFoundError('missing model name', 'OpenaiProvider')
    }

    if (model.startsWith(OpenaiProviderName+'://')) {model = model.slice(OpenaiProviderName.length + 3)}
    options.model = model

    if (!prompt || (!vIsString && !Array.isArray(prompt))) {
      throwError('missing prompt value', 'OpenaiProvider')
    }

    if (options.stop && typeof options.stop === 'string') {
      options.stop = [options.stop]
    }
    if (options.stop_words) {
      let stop_words = options.stop_words || []
      if (typeof options.stop_words === 'string') {
        stop_words.push(options.stop_words)
      }
      if (options.stop) {
        options.stop = [...new Set([...stop_words, ...options.stop])]
      } else {
        options.stop = stop_words
      }
    }

    if (options.response_format?.type === 'json_object' && options.response_format.schema) {
      if (!baseURL || baseURL.includes('api.openai.com')) {
        options.response_format.type = 'json_schema'
      }
      const schema = options.response_format.schema
      if (options.response_format.name) {
        schema.name = options.response_format.name
      } else {
        schema.name = 'result'
      }
      if (options.response_format.description) {
        schema.description = options.response_format.description
      }
      delete options.response_format.description

      if (options.response_format.strict !== undefined) {
        schema.strict = !!options.response_format.strict
      }
      delete options.response_format.strict

      options.response_format.json_schema = {
        schema,
      }
      delete options.response_format.schema
    }

    if (options.response_format?.type === 'json_schema') {
      if (!OpenaiJsonSchemaModels.includes(model)) {
        console.error(`WARN: The json schema response_format is only supported for latest gpt-4o models, not for ${model}.`)
        // throw new NotFoundError('json schema response_format is only supported for latest gpt-4o models', 'OpenaiProvider')
      }
    }

  if (vIsString) {
      prompt = [
        {
          role: 'user',
          content: prompt,
        }
      ]
    }

    options.value = prompt
    return options
  }

  func({model, value, options}: {model: string, value: any, options: AIOptions}): Promise<any> {
    const taskPromise = this.runAsyncCancelableTask(options, async (params: any) => {
      const aborter = params.aborter as TaskAbortController
      const signal = aborter.signal

      params= {...params}
      delete params.aborter
      params = await this.processModelOptions(model, value, params)

      const baseURL = options.apiUrl || this.apiUrl
      const apiKey = options.apiKey || this.apiKey
      const openaiOptions: OpenAIClientOptions = {}
      if (baseURL) {
        openaiOptions.baseURL = baseURL
      }

      if (apiKey) {
        openaiOptions.apiKey = apiKey
      }

      const client = new OpenAI(openaiOptions)

      const chatTemplateId = params.chatTemplateId
      if (chatTemplateId) {delete params.chatTemplateId}

      value = params.value
      delete params.value
      const isStream = options!.stream

      const body: any = {
        ...params,
        // model,
        messages: value,
      }

      // No permission: Uncaught PermissionDeniedError: 403 status code (no body)
      let result: any = await client.chat.completions.create(body, {signal})
      if (result.toReadableStream) {
        console.log('isStream', isStream)
        result = (result as any).toReadableStream().pipeThrough(createOpenaiStreamTransformer())
      } else {
        result = openaiToAIResult(result)
      }

      return result as AIResult
    });

    return taskPromise
  }
}
makeToolFuncCancelable(OpenaiProvider, {asyncFeatures: AsyncFeatures.MultiTask})

export const openai = new OpenaiProvider(OpenaiProviderName)

function createOpenaiStreamTransformer() {
  return new TransformStream({
    transform(chunk: OpenAI.Chat.Completions.ChatCompletionChunk, controller) {
      //
      controller.enqueue(openaiToAIResultChunk(chunk))
    }
  })
}
