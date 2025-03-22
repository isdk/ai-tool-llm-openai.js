[**@isdk/ai-tool-llm-openai**](../README.md)

***

[@isdk/ai-tool-llm-openai](../globals.md) / OpenaiProvider

# Class: OpenaiProvider

Defined in: [packages/ai-tool-llm-openai/src/openai-provider.ts:20](https://github.com/isdk/ai-tool-llm-openai.js/blob/fb4f9970b49723163a84db98a9a8f4b63a264cc1/src/openai-provider.ts#L20)

## Extends

- `CancelableAbility`.`LLMProvider`

## Indexable

\[`name`: `string`\]: `any`

## Constructors

### new OpenaiProvider()

> **new OpenaiProvider**(`name`, `options`?): [`OpenaiProvider`](OpenaiProvider.md)

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:73

#### Parameters

##### name

`string` | `Function` | `FuncItem`

##### options?

`any`

#### Returns

[`OpenaiProvider`](OpenaiProvider.md)

#### Inherited from

`CancelableAbility.constructor`

## Properties

### \_\_task\_aborter

> **\_\_task\_aborter**: `undefined` \| `TaskAbortController` \| `TaskAbortControllers`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:546

#### Inherited from

`CancelableAbility.__task_aborter`

***

### \_\_task\_semaphore

> **\_\_task\_semaphore**: `undefined` \| `Semaphore`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:547

#### Inherited from

`CancelableAbility.__task_semaphore`

***

### \_asyncFeatures?

> `optional` **\_asyncFeatures**: `number`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:538

#### Inherited from

`CancelableAbility._asyncFeatures`

***

### \_isReadyFn?

> `optional` **\_isReadyFn**: `SemaphoreIsReadyFuncType`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:540

#### Inherited from

`CancelableAbility._isReadyFn`

***

### \_maxTaskConcurrency

> **\_maxTaskConcurrency**: `undefined` \| `number`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:539

#### Inherited from

`CancelableAbility._maxTaskConcurrency`

***

### $attributes

> **$attributes**: `Properties`

Defined in: node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/advance.d.ts:5

***

### alias?

> `optional` **alias**: `string` \| `string`[]

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:33

***

### apiKey?

> `optional` **apiKey**: `string`

Defined in: packages/ai-tool-llm/dist/index.d.ts:153

***

### apiUrl?

> `optional` **apiUrl**: `string`

Defined in: packages/ai-tool-llm/dist/index.d.ts:154

***

### cleanMultiTaskAborter()

> **cleanMultiTaskAborter**: (`id`, `aborters`) => `void`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:545

#### Parameters

##### id

`AsyncTaskId`

##### aborters

`TaskAbortControllers`

#### Returns

`void`

#### Inherited from

`CancelableAbility.cleanMultiTaskAborter`

***

### constructor

> **constructor**: `Function`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:125

The initial value of Object.prototype.constructor is the standard built-in Object constructor.

***

### defaultOptions

> **defaultOptions**: `object`

Defined in: node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:74

The default options for export and assign

#### assign?

> `optional` **assign**: `IMergeOptions`

#### export?

> `optional` **export**: `IMergeOptions`

***

### depends

> **depends**: `object`

Defined in: packages/ai-tool-llm/dist/index.d.ts:336

#### Index Signature

\[`name`: `string`\]: `ToolFunc`

***

### description

> **description**: `string`

Defined in: packages/ai-tool-llm/dist/index.d.ts:317

***

### generateAsyncTaskId()

> **generateAsyncTaskId**: (`taskId`?, `aborters`?) => `AsyncTaskId`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:544

#### Parameters

##### taskId?

`AsyncTaskId`

##### aborters?

`TaskAbortControllers`

#### Returns

`AsyncTaskId`

#### Inherited from

`CancelableAbility.generateAsyncTaskId`

***

### isApi?

> `optional` **isApi**: `boolean`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:31

***

### model?

> `optional` **model**: `string`

Defined in: packages/ai-tool-llm/dist/index.d.ts:310

***

### name?

> `optional` **name**: `string`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:25

***

### nonExported1stChar

> **nonExported1stChar**: `string`

Defined in: node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:78

the property with the default prefix '$' will not be exported.

***

### params

> **params**: `object`

Defined in: packages/ai-tool-llm/dist/index.d.ts:318

#### model

> **model**: `object`

##### model.description

> **description**: `string`

##### model.name

> **name**: `string`

##### model.type

> **type**: `string`

#### options

> **options**: `object`

##### options.description

> **description**: `string`

##### options.name

> **name**: `string`

##### options.type

> **type**: `string`

#### value

> **value**: `object`

##### value.description

> **description**: `string`

##### value.name

> **name**: `string`

##### value.type

> **type**: `string`

***

### prior

> **prior**: `number`

Defined in: packages/ai-tool-llm/dist/index.d.ts:151

***

### result

> **result**: `string`

Defined in: packages/ai-tool-llm/dist/index.d.ts:335

***

### rule

> **rule**: (`string` \| `RegExp`)[]

Defined in: [packages/ai-tool-llm-openai/src/openai-provider.ts:22](https://github.com/isdk/ai-tool-llm-openai.js/blob/fb4f9970b49723163a84db98a9a8f4b63a264cc1/src/openai-provider.ts#L22)

***

### scope?

> `optional` **scope**: `any`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:28

***

### setup()?

> `optional` **setup**: (`this`, `options`?) => `void`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:30

#### Parameters

##### this

`ToolFunc`

##### options?

`FuncItem`

#### Returns

`void`

***

### stream?

> `optional` **stream**: `boolean`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:32

***

### supports

> **supports**: `AIModelType` \| `AIModelType`[]

Defined in: packages/ai-tool-llm/dist/index.d.ts:152

***

### tags?

> `optional` **tags**: `string` \| `string`[]

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:29

***

### aliases

> `static` **aliases**: `object`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:54

#### Index Signature

\[`name`: `string`\]: `string`

***

### current?

> `static` `optional` **current**: `string`

Defined in: packages/ai-tool-llm/dist/index.d.ts:313

***

### dataPath

> `static` **dataPath**: `string`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:57

***

### items

> `static` **items**: `object`

Defined in: packages/ai-tool-llm/dist/index.d.ts:314

#### Index Signature

\[`name`: `string`\]: `LLMProvider`

## Accessors

### maxTaskConcurrency

#### Get Signature

> **get** **maxTaskConcurrency**(): `undefined` \| `number`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:548

##### Returns

`undefined` \| `number`

#### Inherited from

`CancelableAbility.maxTaskConcurrency`

***

### semaphore

#### Get Signature

> **get** **semaphore**(): `undefined` \| `Semaphore`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:549

##### Returns

`undefined` \| `Semaphore`

#### Inherited from

`CancelableAbility.semaphore`

## Methods

### \_cleanMultiTaskAborter()

> **\_cleanMultiTaskAborter**(`id`, `aborters`): `void`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:561

#### Parameters

##### id

`AsyncTaskId`

##### aborters

`TaskAbortControllers`

#### Returns

`void`

#### Inherited from

`CancelableAbility._cleanMultiTaskAborter`

***

### \_generateAsyncTaskId()

> **\_generateAsyncTaskId**(`taskId`?, `aborters`?): `AsyncTaskId`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:556

#### Parameters

##### taskId?

`AsyncTaskId`

##### aborters?

`TaskAbortControllers`

#### Returns

`AsyncTaskId`

#### Inherited from

`CancelableAbility._generateAsyncTaskId`

***

### $cleanMultiTaskAborter()

> **$cleanMultiTaskAborter**(`id`, `aborters`): `void`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:559

#### Parameters

##### id

`AsyncTaskId`

##### aborters

`TaskAbortControllers`

#### Returns

`void`

#### Inherited from

`CancelableAbility.$cleanMultiTaskAborter`

***

### $generateAsyncTaskId()

> **$generateAsyncTaskId**(`taskId`?, `aborters`?): `undefined` \| `AsyncTaskId`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:557

#### Parameters

##### taskId?

`AsyncTaskId`

##### aborters?

`TaskAbortControllers`

#### Returns

`undefined` \| `AsyncTaskId`

#### Inherited from

`CancelableAbility.$generateAsyncTaskId`

***

### abort()

> **abort**(`reason`?, `data`?): `void`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:571

#### Parameters

##### reason?

`string`

##### data?

`any`

#### Returns

`void`

#### Inherited from

`CancelableAbility.abort`

***

### addModel()?

> `optional` **addModel**(`modelName`, `model`): `Promise`\<`any`\>

Defined in: packages/ai-tool-llm/dist/index.d.ts:308

#### Parameters

##### modelName

`string`

##### model

`any`

#### Returns

`Promise`\<`any`\>

***

### arr2ObjParams()

> **arr2ObjParams**(`params`): `any`[]

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:76

#### Parameters

##### params

`any`[]

#### Returns

`any`[]

***

### assign()

> **assign**(`src`, `options`?): `this`

Defined in: node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:106

Assign the values from the src object.

#### Parameters

##### src

`any`

the source object

##### options?

`IMergeOptions`

#### Returns

`this`

this object

***

### assignProperty()

> **assignProperty**(`src`, `name`, `value`, `attrs`?, `options`?): `void`

Defined in: node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:117

Assign a property of src to this object.

#### Parameters

##### src

`any`

the src object

##### name

`string`

the property name to assign

##### value

`any`

the property value to assign

##### attrs?

`any`

the attributes object

##### options?

`IMergeOptions`

#### Returns

`void`

***

### assignPropertyTo()

> `abstract` **assignPropertyTo**(`dest`, `src`, `name`, `value`, `attrs`?, `options`?): `void`

Defined in: node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:131

Assign the property value from the src to destination object.

#### Parameters

##### dest

`any`

The destination object

##### src

`any`

The src object

##### name

`string`

The property name

##### value

`any`

The property value

##### attrs?

`any`

The attributes object of the property

##### options?

`IMergeOptions`

#### Returns

`void`

***

### assignTo()

> **assignTo**(`dest`?, `options`?): `any`

Defined in: node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:191

Assign this attributes to the dest object

#### Parameters

##### dest?

`any`

the destination object

##### options?

`IMergeOptions`

#### Returns

`any`

the dest object

***

### cleanTaskAborter()

> **cleanTaskAborter**(`aborter`): `void`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:560

#### Parameters

##### aborter

`TaskAbortController`

#### Returns

`void`

#### Inherited from

`CancelableAbility.cleanTaskAborter`

***

### clone()

> **clone**(`options`?): `any`

Defined in: node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:155

Create a new object with the same values of attributes.

#### Parameters

##### options?

`IMergeOptions`

#### Returns

`any`

the new object

***

### cloneTo()

> **cloneTo**(`dest`, `options`?): `any`

Defined in: node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:148

Create and assign the values to the destination object.

#### Parameters

##### dest

`any`

the destination object

##### options?

`IMergeOptions`

#### Returns

`any`

the new dest object

***

### countTokens()

> **countTokens**(`text`, `options`?): `Promise`\<`number`\>

Defined in: packages/ai-tool-llm/dist/index.d.ts:369

#### Parameters

##### text

`string` | `AIChatMessageParam`[]

##### options?

`AITokenizeOptions`

#### Returns

`Promise`\<`number`\>

***

### createAborter()

> **createAborter**(`params`?, `taskId`?, `raiseError`?): `TaskAbortController`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:558

#### Parameters

##### params?

`any`

##### taskId?

`AsyncTaskId`

##### raiseError?

`boolean`

#### Returns

`TaskAbortController`

#### Inherited from

`CancelableAbility.createAborter`

***

### createTaskPromise()

> **createTaskPromise**\<`Output`\>(`runTask`, `params`, `options`?): `TaskPromise`\<`Output`\>

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:562

#### Type Parameters

• **Output** = `any`

#### Parameters

##### runTask

(`params`, `aborter`) => `Promise`\<`Output`\>

##### params

`Record`\<`string`, `any`\>

##### options?

###### raiseError?

`boolean`

###### taskId?

`AsyncTaskId`

#### Returns

`TaskPromise`\<`Output`\>

#### Inherited from

`CancelableAbility.createTaskPromise`

***

### defineProperties()

> `abstract` **defineProperties**(`aProperties`): `any`

Defined in: node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:89

Define the attributes of this object.

#### Parameters

##### aProperties

`SimplePropDescriptors`

the defined attributes of the object

#### Returns

`any`

***

### deleteModel()?

> `optional` **deleteModel**(`modelName`): `Promise`\<`any`\>

Defined in: packages/ai-tool-llm/dist/index.d.ts:309

#### Parameters

##### modelName

`string`

#### Returns

`Promise`\<`any`\>

***

### exportTo()

> **exportTo**(`dest`, `options`?): `any`

Defined in: node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:173

Export attributes to the dest json object.

#### Parameters

##### dest

`any`

the destination object

##### options?

`IExportOptions`

#### Returns

`any`

the dest object.

***

### formatPrompt()

> **formatPrompt**(`messages`, `modelInfo`?, `options`?): `Promise`\<`undefined` \| `string`\>

Defined in: packages/ai-tool-llm/dist/index.d.ts:359

#### Parameters

##### messages

`AIChatMessageParam`[]

##### modelInfo?

`string` | `AIModelParams`

##### options?

###### add_generation_prompt?

`boolean`

###### chatTemplate?

`AIPromptResult`

###### defaultTemplate?

`boolean`

###### prompt?

`any`

###### SystemTemplate?

`AIPromptResult`

###### type?

`AIPromptType`

#### Returns

`Promise`\<`undefined` \| `string`\>

***

### func()

> **func**(`__namedParameters`): `Promise`\<`any`\>

Defined in: [packages/ai-tool-llm-openai/src/openai-provider.ts:111](https://github.com/isdk/ai-tool-llm-openai.js/blob/fb4f9970b49723163a84db98a9a8f4b63a264cc1/src/openai-provider.ts#L111)

#### Parameters

##### \_\_namedParameters

###### model

`string`

###### options

`AIOptions`

###### value

`any`

#### Returns

`Promise`\<`any`\>

***

### getChatTemplate()

> **getChatTemplate**(`modelInfo`?, `options`?): `Promise`\<`undefined` \| `AIPromptResult`\>

Defined in: packages/ai-tool-llm/dist/index.d.ts:354

#### Parameters

##### modelInfo?

`string` | `AIModelParams`

##### options?

###### defaultTemplate?

`boolean`

###### provider?

`string`

###### type?

`AIPromptType`

#### Returns

`Promise`\<`undefined` \| `AIPromptResult`\>

***

### getCurrentProvider()

> **getCurrentProvider**(): `undefined` \| `LLMProvider`

Defined in: packages/ai-tool-llm/dist/index.d.ts:351

#### Returns

`undefined` \| `LLMProvider`

***

### getDefaultParameters()

> **getDefaultParameters**(`chatTemplate`, `model`): `any`

Defined in: packages/ai-tool-llm/dist/index.d.ts:367

#### Parameters

##### chatTemplate

`AIPromptSettings`

##### model

`string`

#### Returns

`any`

***

### getFunc()

> **getFunc**(`name`?): `any`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:82

#### Parameters

##### name?

`string`

#### Returns

`any`

***

### getFuncWithPos()

> **getFuncWithPos**(`name`?): `any`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:87

#### Parameters

##### name?

`string`

#### Returns

`any`

***

### getModelInfo()

> **getModelInfo**(`modelName`?, `options`?): `Promise`\<`undefined` \| `AIModelParams`\>

Defined in: packages/ai-tool-llm/dist/index.d.ts:353

#### Parameters

##### modelName?

`string`

##### options?

`any`

#### Returns

`Promise`\<`undefined` \| `AIModelParams`\>

***

### getProperties()

> `abstract` **getProperties**(): `PropDescriptors`

Defined in: node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:98

Get the defined attributes.

#### Returns

`PropDescriptors`

the descriptors of properties object

***

### getProvider()

> **getProvider**(`name`): `ToolFunc`

Defined in: packages/ai-tool-llm/dist/index.d.ts:350

#### Parameters

##### name

`string`

#### Returns

`ToolFunc`

***

### getRunningTask()

> **getRunningTask**(`taskId`?): `undefined` \| `TaskAbortController`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:554

#### Parameters

##### taskId?

`AsyncTaskId`

#### Returns

`undefined` \| `TaskAbortController`

#### Inherited from

`CancelableAbility.getRunningTask`

***

### getRunningTaskCount()

> **getRunningTaskCount**(): `number`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:555

#### Returns

`number`

#### Inherited from

`CancelableAbility.getRunningTaskCount`

***

### getSemaphore()

> **getSemaphore**(`isReadyFn`?): `undefined` \| `Semaphore`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:550

#### Parameters

##### isReadyFn?

`SemaphoreIsReadyFuncType`

#### Returns

`undefined` \| `Semaphore`

#### Inherited from

`CancelableAbility.getSemaphore`

***

### hasAsyncFeature()

> **hasAsyncFeature**(`feature`): `boolean`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:88

#### Parameters

##### feature

`AsyncFeatureBits`

#### Returns

`boolean`

#### Inherited from

`CancelableAbility.hasAsyncFeature`

***

### hasOwnProperty()

> **hasOwnProperty**(`v`): `boolean`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:140

Determines whether an object has a property with the specified name.

#### Parameters

##### v

`PropertyKey`

A property name.

#### Returns

`boolean`

***

### initialize()

> **initialize**(`src`?): `this`

Defined in: node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:139

Initialize object and assign attribute values from src if src exists.

#### Parameters

##### src?

`any`

#### Returns

`this`

this object.

***

### isAborted()

> **isAborted**(`taskId`?): `boolean`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:553

#### Parameters

##### taskId?

`AsyncTaskId`

#### Returns

`boolean`

#### Inherited from

`CancelableAbility.isAborted`

***

### isModelNameMatched()

> **isModelNameMatched**(`modelName`, `rule`?): `undefined` \| `string` \| `RegExpExecArray`

Defined in: packages/ai-tool-llm/dist/index.d.ts:341

#### Parameters

##### modelName

`string`

##### rule?

`AIModelNameRules`

#### Returns

`undefined` \| `string` \| `RegExpExecArray`

***

### isPrototypeOf()

> **isPrototypeOf**(`v`): `boolean`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:146

Determines whether an object exists in another object's prototype chain.

#### Parameters

##### v

`Object`

Another object whose prototype chain is to be checked.

#### Returns

`boolean`

***

### isSame()

> **isSame**(`src`, `options`?): `boolean`

Defined in: node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:200

Check the src object whether “equals” this object.

#### Parameters

##### src

`any`

The source object

##### options?

`IMergeOptions`

#### Returns

`boolean`

***

### isStream()

> **isStream**(`params`): `any`

Defined in: packages/ai-tool-llm/dist/index.d.ts:342

#### Parameters

##### params

`ServerFuncParams`

#### Returns

`any`

***

### listModels()?

> `optional` **listModels**(): `Promise`\<`undefined` \| `string`[]\>

Defined in: packages/ai-tool-llm/dist/index.d.ts:306

#### Returns

`Promise`\<`undefined` \| `string`[]\>

***

### listProviders()

> **listProviders**(`options`?): `object`

Defined in: packages/ai-tool-llm/dist/index.d.ts:344

#### Parameters

##### options?

###### all?

`boolean`

###### filter?

`AIModelNameRules`

#### Returns

`object`

***

### mergeTo()

> **mergeTo**(`dest`, `options`?): `any`

Defined in: node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:164

Merge this attributes to dest object.

#### Parameters

##### dest

`any`

The destination object

##### options?

`IMergeOptions`

#### Returns

`any`

the dest object.

***

### obj2ArrParams()

> **obj2ArrParams**(`params`?): `any`[]

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:77

#### Parameters

##### params?

`any`

#### Returns

`any`[]

***

### processModelOptions()

> **processModelOptions**(`model`, `prompt`, `options`): `Promise`\<`AIOptions`\>

Defined in: [packages/ai-tool-llm-openai/src/openai-provider.ts:32](https://github.com/isdk/ai-tool-llm-openai.js/blob/fb4f9970b49723163a84db98a9a8f4b63a264cc1/src/openai-provider.ts#L32)

#### Parameters

##### model

`string`

##### prompt

`any`

##### options

`AIOptions`

#### Returns

`Promise`\<`AIOptions`\>

***

### propertyIsEnumerable()

> **propertyIsEnumerable**(`v`): `boolean`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:152

Determines whether a specified property is enumerable.

#### Parameters

##### v

`PropertyKey`

A property name.

#### Returns

`boolean`

***

### register()

> **register**(): `boolean` \| `ToolFunc`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:74

#### Returns

`boolean` \| `ToolFunc`

***

### run()

> **run**(`params`?): `Promise`\<`any`\>

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:79

#### Parameters

##### params?

`any`

#### Returns

`Promise`\<`any`\>

***

### runAs()

> **runAs**(`name`, `params`?): `Promise`\<`any`\>

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:80

#### Parameters

##### name

`string`

##### params?

`any`

#### Returns

`Promise`\<`any`\>

***

### runAsSync()

> **runAsSync**(`name`, `params`?): `any`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:81

#### Parameters

##### name

`string`

##### params?

`any`

#### Returns

`any`

***

### runAsyncCancelableTask()

> **runAsyncCancelableTask**\<`Output`\>(`params`, `runTask`, `options`?): `TaskPromise`\<`Output`\>

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:566

#### Type Parameters

• **Output** = `any`

#### Parameters

##### params

`undefined` | `Record`\<`string`, `any`\>

##### runTask

(`params`, `aborter`) => `Promise`\<`Output`\>

##### options?

###### isReadyFn?

`SemaphoreIsReadyFuncType`

###### raiseError?

`boolean`

###### taskId?

`AsyncTaskId`

#### Returns

`TaskPromise`\<`Output`\>

#### Inherited from

`CancelableAbility.runAsyncCancelableTask`

***

### runSync()

> **runSync**(`params`?): `any`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:78

#### Parameters

##### params?

`any`

#### Returns

`any`

***

### runWithPos()

> **runWithPos**(...`params`): `Promise`\<`any`\>

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:85

#### Parameters

##### params

...`any`[]

#### Returns

`Promise`\<`any`\>

***

### runWithPosAs()

> **runWithPosAs**(`name`, ...`params`): `Promise`\<`any`\>

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:86

#### Parameters

##### name

`string`

##### params

...`any`[]

#### Returns

`Promise`\<`any`\>

***

### runWithPosAsSync()

> **runWithPosAsSync**(`name`, ...`params`): `any`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:84

#### Parameters

##### name

`string`

##### params

...`any`[]

#### Returns

`any`

***

### runWithPosSync()

> **runWithPosSync**(...`params`): `any`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:83

#### Parameters

##### params

...`any`[]

#### Returns

`any`

***

### setCurrentProvider()

> **setCurrentProvider**(`name`): `void`

Defined in: packages/ai-tool-llm/dist/index.d.ts:352

#### Parameters

##### name

`string`

#### Returns

`void`

***

### toJSON()

> **toJSON**(): `any`

Defined in: node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:182

#### Returns

`any`

***

### tokenize()

> **tokenize**(`text`, `options`?): `Promise`\<`number`[]\>

Defined in: packages/ai-tool-llm/dist/index.d.ts:368

#### Parameters

##### text

`string` | `AIChatMessageParam`[]

##### options?

`AITokenizeOptions`

#### Returns

`Promise`\<`number`[]\>

***

### toLocaleString()

> **toLocaleString**(): `string`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:131

Returns a date converted to a string using the current locale.

#### Returns

`string`

***

### toObject()

> **toObject**(`options`?): `any`

Defined in: node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:181

Convert the attributes to the json object

#### Parameters

##### options?

`any`

#### Returns

`any`

the json object.

***

### toString()

> **toString**(): `string`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:128

Returns a string representation of an object.

#### Returns

`string`

***

### unregister()

> **unregister**(): `any`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:75

#### Returns

`any`

***

### updateModel()?

> `optional` **updateModel**(`modelName`, `model`): `Promise`\<`any`\>

Defined in: packages/ai-tool-llm/dist/index.d.ts:307

#### Parameters

##### modelName

`string`

##### model

`any`

#### Returns

`Promise`\<`any`\>

***

### valueOf()

> **valueOf**(): `Object`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:134

Returns the primitive value of the specified object.

#### Returns

`Object`

***

### assign()

#### Call Signature

> `static` **assign**\<`T`, `U`\>(`target`, `source`): `T` & `U`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2015.core.d.ts:286

Copy the values of all of the enumerable own properties from one or more source objects to a
target object. Returns the target object.

##### Type Parameters

• **T** *extends* `object`

• **U**

##### Parameters

###### target

`T`

The target object to copy to.

###### source

`U`

The source object from which to copy properties.

##### Returns

`T` & `U`

#### Call Signature

> `static` **assign**\<`T`, `U`, `V`\>(`target`, `source1`, `source2`): `T` & `U` & `V`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2015.core.d.ts:295

Copy the values of all of the enumerable own properties from one or more source objects to a
target object. Returns the target object.

##### Type Parameters

• **T** *extends* `object`

• **U**

• **V**

##### Parameters

###### target

`T`

The target object to copy to.

###### source1

`U`

The first source object from which to copy properties.

###### source2

`V`

The second source object from which to copy properties.

##### Returns

`T` & `U` & `V`

#### Call Signature

> `static` **assign**\<`T`, `U`, `V`, `W`\>(`target`, `source1`, `source2`, `source3`): `T` & `U` & `V` & `W`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2015.core.d.ts:305

Copy the values of all of the enumerable own properties from one or more source objects to a
target object. Returns the target object.

##### Type Parameters

• **T** *extends* `object`

• **U**

• **V**

• **W**

##### Parameters

###### target

`T`

The target object to copy to.

###### source1

`U`

The first source object from which to copy properties.

###### source2

`V`

The second source object from which to copy properties.

###### source3

`W`

The third source object from which to copy properties.

##### Returns

`T` & `U` & `V` & `W`

#### Call Signature

> `static` **assign**(`target`, ...`sources`): `any`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2015.core.d.ts:313

Copy the values of all of the enumerable own properties from one or more source objects to a
target object. Returns the target object.

##### Parameters

###### target

`object`

The target object to copy to.

###### sources

...`any`[]

One or more source objects from which to copy properties

##### Returns

`any`

***

### create()

#### Call Signature

> `static` **create**(`o`): `any`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:188

Creates an object that has the specified prototype or that has null prototype.

##### Parameters

###### o

Object to use as a prototype. May be null.

`null` | `object`

##### Returns

`any`

#### Call Signature

> `static` **create**(`o`, `properties`): `any`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:195

Creates an object that has the specified prototype, and that optionally contains specified properties.

##### Parameters

###### o

Object to use as a prototype. May be null

`null` | `object`

###### properties

`PropertyDescriptorMap` & `ThisType`\<`any`\>

JavaScript object that contains one or more property descriptors.

##### Returns

`any`

***

### defineProperties()

> `static` **defineProperties**(`aTarget`, `aProperties`, `recreate`?): `any`

Defined in: node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/advance.d.ts:11

Adds one or more properties to an object, and/or modifies attributes of existing properties.

#### Parameters

##### aTarget

`any`

##### aProperties

`PropDescriptors`

##### recreate?

`boolean`

#### Returns

`any`

***

### defineProperty()

> `static` **defineProperty**\<`T`\>(`o`, `p`, `attributes`): `T`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:203

Adds a property to an object, or modifies attributes of an existing property.

#### Type Parameters

• **T**

#### Parameters

##### o

`T`

Object on which to add or modify the property. This can be a native JavaScript object (that is, a user-defined object or a built in object) or a DOM object.

##### p

`PropertyKey`

The property name.

##### attributes

`PropertyDescriptor` & `ThisType`\<`any`\>

Descriptor for the property. It can be for a data property or an accessor property.

#### Returns

`T`

***

### entries()

#### Call Signature

> `static` **entries**\<`T`\>(`o`): \[`string`, `T`\][]

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2017.object.d.ts:36

Returns an array of key/values of the enumerable own properties of an object

##### Type Parameters

• **T**

##### Parameters

###### o

Object that contains the properties and methods. This can be an object that you created or an existing Document Object Model (DOM) object.

\{\} | `ArrayLike`\<`T`\>

##### Returns

\[`string`, `T`\][]

#### Call Signature

> `static` **entries**(`o`): \[`string`, `any`\][]

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2017.object.d.ts:42

Returns an array of key/values of the enumerable own properties of an object

##### Parameters

###### o

Object that contains the properties and methods. This can be an object that you created or an existing Document Object Model (DOM) object.

##### Returns

\[`string`, `any`\][]

***

### freeze()

#### Call Signature

> `static` **freeze**\<`T`\>(`f`): `T`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:222

Prevents the modification of existing property attributes and values, and prevents the addition of new properties.

##### Type Parameters

• **T** *extends* `Function`

##### Parameters

###### f

`T`

Object on which to lock the attributes.

##### Returns

`T`

#### Call Signature

> `static` **freeze**\<`T`, `U`\>(`o`): `Readonly`\<`T`\>

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:228

Prevents the modification of existing property attributes and values, and prevents the addition of new properties.

##### Type Parameters

• **T** *extends* `object`

• **U** *extends* `string` \| `number` \| `bigint` \| `boolean` \| `symbol`

##### Parameters

###### o

`T`

Object on which to lock the attributes.

##### Returns

`Readonly`\<`T`\>

#### Call Signature

> `static` **freeze**\<`T`\>(`o`): `Readonly`\<`T`\>

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:234

Prevents the modification of existing property attributes and values, and prevents the addition of new properties.

##### Type Parameters

• **T**

##### Parameters

###### o

`T`

Object on which to lock the attributes.

##### Returns

`Readonly`\<`T`\>

***

### fromEntries()

#### Call Signature

> `static` **fromEntries**\<`T`\>(`entries`): `object`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2019.object.d.ts:26

Returns an object created by key-value entries for properties and methods

##### Type Parameters

• **T** = `any`

##### Parameters

###### entries

`Iterable`\<readonly \[`PropertyKey`, `T`\]\>

An iterable object that contains key-value entries for properties and methods.

##### Returns

`object`

#### Call Signature

> `static` **fromEntries**(`entries`): `any`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2019.object.d.ts:32

Returns an object created by key-value entries for properties and methods

##### Parameters

###### entries

`Iterable`\<readonly `any`[]\>

An iterable object that contains key-value entries for properties and methods.

##### Returns

`any`

***

### get()

> `static` **get**(`name`): `ToolFunc`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:58

#### Parameters

##### name

`string`

#### Returns

`ToolFunc`

***

### getAllByTag()

> `static` **getAllByTag**(`tagName`): `ToolFunc`[]

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:61

#### Parameters

##### tagName

`string`

#### Returns

`ToolFunc`[]

***

### getByModel()

> `static` **getByModel**(`modelName`?): `undefined` \| `LLMProvider`

Defined in: packages/ai-tool-llm/dist/index.d.ts:339

#### Parameters

##### modelName?

`string`

#### Returns

`undefined` \| `LLMProvider`

***

### getByTag()

> `static` **getByTag**(`tagName`): `undefined` \| `ToolFunc`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:60

#### Parameters

##### tagName

`string`

#### Returns

`undefined` \| `ToolFunc`

***

### getCurrentProvider()

> `static` **getCurrentProvider**(): `undefined` \| `LLMProvider`

Defined in: packages/ai-tool-llm/dist/index.d.ts:340

#### Returns

`undefined` \| `LLMProvider`

***

### getFunc()

> `static` **getFunc**(`name`): `any`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:65

#### Parameters

##### name

`string`

#### Returns

`any`

***

### getFuncWithPos()

> `static` **getFuncWithPos**(`name`): `any`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:68

#### Parameters

##### name

`string`

#### Returns

`any`

***

### getOwnPropertyDescriptor()

> `static` **getOwnPropertyDescriptor**(`o`, `p`): `undefined` \| `PropertyDescriptor`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:175

Gets the own property descriptor of the specified object.
An own property descriptor is one that is defined directly on the object and is not inherited from the object's prototype.

#### Parameters

##### o

`any`

Object that contains the property.

##### p

`PropertyKey`

Name of the property.

#### Returns

`undefined` \| `PropertyDescriptor`

***

### getOwnPropertyDescriptors()

> `static` **getOwnPropertyDescriptors**\<`T`\>(`o`): \{ \[P in string \| number \| symbol\]: TypedPropertyDescriptor\<T\[P\]\> \} & `object`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2017.object.d.ts:48

Returns an object containing all own property descriptors of an object

#### Type Parameters

• **T**

#### Parameters

##### o

`T`

Object that contains the properties and methods. This can be an object that you created or an existing Document Object Model (DOM) object.

#### Returns

\{ \[P in string \| number \| symbol\]: TypedPropertyDescriptor\<T\[P\]\> \} & `object`

***

### getOwnPropertyNames()

> `static` **getOwnPropertyNames**(`o`): `string`[]

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:182

Returns the names of the own properties of an object. The own properties of an object are those that are defined directly
on that object, and are not inherited from the object's prototype. The properties of an object include both fields (objects) and functions.

#### Parameters

##### o

`any`

Object that contains the own properties.

#### Returns

`string`[]

***

### getOwnPropertySymbols()

> `static` **getOwnPropertySymbols**(`o`): `symbol`[]

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2015.core.d.ts:319

Returns an array of all symbol properties found directly on object o.

#### Parameters

##### o

`any`

Object to retrieve the symbols from.

#### Returns

`symbol`[]

***

### getProperties()

> `static` **getProperties**(): `PropDescriptors`

Defined in: node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/advance.d.ts:10

get all properties descriptor include inherited.

#### Returns

`PropDescriptors`

***

### getPrototypeOf()

> `static` **getPrototypeOf**(`o`): `any`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:167

Returns the prototype of an object.

#### Parameters

##### o

`any`

The object that references the prototype.

#### Returns

`any`

***

### hasAsyncFeature()

> `static` **hasAsyncFeature**(`feature`): `boolean`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:62

#### Parameters

##### feature

`AsyncFeatureBits`

#### Returns

`boolean`

#### Inherited from

`CancelableAbility.hasAsyncFeature`

***

### is()

> `static` **is**(`value1`, `value2`): `boolean`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2015.core.d.ts:332

Returns true if the values are the same value, false otherwise.

#### Parameters

##### value1

`any`

The first value.

##### value2

`any`

The second value.

#### Returns

`boolean`

***

### isExtensible()

> `static` **isExtensible**(`o`): `boolean`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:258

Returns a value that indicates whether new properties can be added to an object.

#### Parameters

##### o

`any`

Object to test.

#### Returns

`boolean`

***

### isFrozen()

> `static` **isFrozen**(`o`): `boolean`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:252

Returns true if existing property attributes and values cannot be modified in an object, and new properties cannot be added to the object.

#### Parameters

##### o

`any`

Object to test.

#### Returns

`boolean`

***

### isSealed()

> `static` **isSealed**(`o`): `boolean`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:246

Returns true if existing property attributes cannot be modified in an object and new properties cannot be added to the object.

#### Parameters

##### o

`any`

Object to test.

#### Returns

`boolean`

***

### keys()

#### Call Signature

> `static` **keys**(`o`): `string`[]

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:264

Returns the names of the enumerable string properties and methods of an object.

##### Parameters

###### o

`object`

Object that contains the properties and methods. This can be an object that you created or an existing Document Object Model (DOM) object.

##### Returns

`string`[]

#### Call Signature

> `static` **keys**(`o`): `string`[]

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2015.core.d.ts:325

Returns the names of the enumerable string properties and methods of an object.

##### Parameters

###### o

Object that contains the properties and methods. This can be an object that you created or an existing Document Object Model (DOM) object.

##### Returns

`string`[]

***

### list()

> `static` **list**(): `Funcs`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:59

#### Returns

`Funcs`

***

### preventExtensions()

> `static` **preventExtensions**\<`T`\>(`o`): `T`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:240

Prevents the addition of new properties to an object.

#### Type Parameters

• **T**

#### Parameters

##### o

`T`

Object to make non-extensible.

#### Returns

`T`

***

### register()

#### Call Signature

> `static` **register**(`name`, `options`): `boolean` \| `ToolFunc`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:69

##### Parameters

###### name

`string`

###### options

`FuncItem`

##### Returns

`boolean` \| `ToolFunc`

#### Call Signature

> `static` **register**(`func`, `options`): `boolean` \| `ToolFunc`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:70

##### Parameters

###### func

`Function`

###### options

`FuncItem`

##### Returns

`boolean` \| `ToolFunc`

#### Call Signature

> `static` **register**(`name`, `options`?): `boolean` \| `ToolFunc`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:71

##### Parameters

###### name

`string` | `Function` | `ToolFunc` | `FuncItem`

###### options?

`FuncItem`

##### Returns

`boolean` \| `ToolFunc`

***

### run()

> `static` **run**(`name`, `params`?): `Promise`\<`any`\>

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:63

#### Parameters

##### name

`string`

##### params?

`any`

#### Returns

`Promise`\<`any`\>

***

### runSync()

> `static` **runSync**(`name`, `params`?): `any`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:64

#### Parameters

##### name

`string`

##### params?

`any`

#### Returns

`any`

***

### runWithPos()

> `static` **runWithPos**(`name`, ...`params`): `Promise`\<`any`\>

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:66

#### Parameters

##### name

`string`

##### params

...`any`[]

#### Returns

`Promise`\<`any`\>

***

### runWithPosSync()

> `static` **runWithPosSync**(`name`, ...`params`): `any`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:67

#### Parameters

##### name

`string`

##### params

...`any`[]

#### Returns

`any`

***

### seal()

> `static` **seal**\<`T`\>(`o`): `T`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:216

Prevents the modification of attributes of existing properties, and prevents the addition of new properties.

#### Type Parameters

• **T**

#### Parameters

##### o

`T`

Object on which to lock the attributes.

#### Returns

`T`

***

### setPrototypeOf()

> `static` **setPrototypeOf**(`o`, `proto`): `any`

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2015.core.d.ts:339

Sets the prototype of a specified object o to object proto or null. Returns the object o.

#### Parameters

##### o

`any`

The object to change its prototype.

##### proto

The value of the new prototype or null.

`null` | `object`

#### Returns

`any`

***

### unregister()

> `static` **unregister**(`name`): `undefined` \| `ToolFunc`

Defined in: packages/ai-tool/dist/index-BkwhCK1q.d.ts:72

#### Parameters

##### name

`string`

#### Returns

`undefined` \| `ToolFunc`

***

### values()

#### Call Signature

> `static` **values**\<`T`\>(`o`): `T`[]

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2017.object.d.ts:24

Returns an array of values of the enumerable own properties of an object

##### Type Parameters

• **T**

##### Parameters

###### o

Object that contains the properties and methods. This can be an object that you created or an existing Document Object Model (DOM) object.

\{\} | `ArrayLike`\<`T`\>

##### Returns

`T`[]

#### Call Signature

> `static` **values**(`o`): `any`[]

Defined in: node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2017.object.d.ts:30

Returns an array of values of the enumerable own properties of an object

##### Parameters

###### o

Object that contains the properties and methods. This can be an object that you created or an existing Document Object Model (DOM) object.

##### Returns

`any`[]
