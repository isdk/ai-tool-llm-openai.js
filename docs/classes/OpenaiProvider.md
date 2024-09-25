[**@isdk/ai-tool-llm-openai**](../README.md) • **Docs**

***

[@isdk/ai-tool-llm-openai](../globals.md) / OpenaiProvider

# Class: OpenaiProvider

## Extends

- `CancelableAbility`.`LLMProvider`

## Constructors

### new OpenaiProvider()

> **new OpenaiProvider**(`name`, `options`?): [`OpenaiProvider`](OpenaiProvider.md)

#### Parameters

• **name**: `string` \| `Function` \| `FuncItem`

• **options?**: `any`

#### Returns

[`OpenaiProvider`](OpenaiProvider.md)

#### Inherited from

`CancelableAbility.constructor`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:80

## Properties

### \_\_task\_aborter

> **\_\_task\_aborter**: `undefined` \| `TaskAbortController` \| `TaskAbortControllers`

#### Inherited from

`CancelableAbility.__task_aborter`

#### Defined in

packages/ai-tool/dist/index.d.ts:212

***

### \_\_task\_semaphore

> **\_\_task\_semaphore**: `undefined` \| `Semaphore`

#### Inherited from

`CancelableAbility.__task_semaphore`

#### Defined in

packages/ai-tool/dist/index.d.ts:213

***

### \_asyncFeatures?

> `optional` **\_asyncFeatures**: `number`

#### Inherited from

`CancelableAbility._asyncFeatures`

#### Defined in

packages/ai-tool/dist/index.d.ts:205

***

### \_maxTaskConcurrency

> **\_maxTaskConcurrency**: `undefined` \| `number`

#### Inherited from

`CancelableAbility._maxTaskConcurrency`

#### Defined in

packages/ai-tool/dist/index.d.ts:206

***

### $attributes

> **$attributes**: `Properties`

#### Defined in

node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/advance.d.ts:5

***

### apiKey?

> `optional` **apiKey**: `string`

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:154

***

### apiUrl?

> `optional` **apiUrl**: `string`

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:155

***

### cleanMultiTaskAborter()

> **cleanMultiTaskAborter**: (`id`, `aborters`) => `void`

#### Parameters

• **id**: `AsyncTaskId`

• **aborters**: `TaskAbortControllers`

#### Returns

`void`

#### Inherited from

`CancelableAbility.cleanMultiTaskAborter`

#### Defined in

packages/ai-tool/dist/index.d.ts:211

***

### constructor

> **constructor**: `Function`

The initial value of Object.prototype.constructor is the standard built-in Object constructor.

#### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es5.d.ts:125

***

### defaultOptions

> **defaultOptions**: `object`

The default options for export and assign

#### assign?

> `optional` **assign**: `IMergeOptions`

#### export?

> `optional` **export**: `IMergeOptions`

#### Defined in

node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:74

***

### depends

> **depends**: `object`

#### Index Signature

 \[`name`: `string`\]: `ToolFunc`

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:331

***

### description

> **description**: `string`

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:312

***

### generateAsyncTaskId()

> **generateAsyncTaskId**: (`taskId`?, `aborters`?) => `AsyncTaskId`

#### Parameters

• **taskId?**: `AsyncTaskId`

• **aborters?**: `TaskAbortControllers`

#### Returns

`AsyncTaskId`

#### Inherited from

`CancelableAbility.generateAsyncTaskId`

#### Defined in

packages/ai-tool/dist/index.d.ts:210

***

### isApi?

> `optional` **isApi**: `boolean`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:42

***

### model?

> `optional` **model**: `string`

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:305

***

### name?

> `optional` **name**: `string`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:36

***

### nonExported1stChar

> **nonExported1stChar**: `string`

the property with the default prefix '$' will not be exported.

#### Defined in

node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:78

***

### params

> **params**: `object`

#### model

> **model**: `object`

#### model.description

> **description**: `string`

#### model.name

> **name**: `string`

#### model.type

> **type**: `string`

#### options

> **options**: `object`

#### options.description

> **description**: `string`

#### options.name

> **name**: `string`

#### options.type

> **type**: `string`

#### value

> **value**: `object`

#### value.description

> **description**: `string`

#### value.name

> **name**: `string`

#### value.type

> **type**: `string`

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:313

***

### prior

> **prior**: `number`

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:152

***

### result

> **result**: `string`

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:330

***

### rule

> **rule**: (`string` \| `RegExp`)[]

#### Defined in

[packages/ai-tool-llm-openai/src/openai-provider.ts:22](https://github.com/isdk/ai-tool-llm-openai.js/blob/3fbd49844808b871a953477e899e1386a8cded30/src/openai-provider.ts#L22)

***

### scope?

> `optional` **scope**: `any`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:39

***

### setup()?

> `optional` **setup**: (`this`, `options`?) => `void`

#### Parameters

• **this**: `ToolFunc`

• **options?**: `FuncItem`

#### Returns

`void`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:41

***

### stream?

> `optional` **stream**: `boolean`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:43

***

### supports

> **supports**: `AIModelType` \| `AIModelType`[]

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:153

***

### tags?

> `optional` **tags**: `string` \| `string`[]

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:40

***

### current?

> `static` `optional` **current**: `string`

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:308

***

### dataPath

> `static` **dataPath**: `string`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:64

***

### items

> `static` **items**: `object`

#### Index Signature

 \[`name`: `string`\]: `LLMProvider`

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:309

## Accessors

### maxTaskConcurrency

> `get` **maxTaskConcurrency**(): `undefined` \| `number`

#### Returns

`undefined` \| `number`

#### Inherited from

`CancelableAbility.maxTaskConcurrency`

#### Defined in

packages/ai-tool/dist/index.d.ts:214

***

### semaphore

> `get` **semaphore**(): `undefined` \| `Semaphore`

#### Returns

`undefined` \| `Semaphore`

#### Inherited from

`CancelableAbility.semaphore`

#### Defined in

packages/ai-tool/dist/index.d.ts:215

## Methods

### \_cleanMultiTaskAborter()

> **\_cleanMultiTaskAborter**(`id`, `aborters`): `void`

#### Parameters

• **id**: `AsyncTaskId`

• **aborters**: `TaskAbortControllers`

#### Returns

`void`

#### Inherited from

`CancelableAbility._cleanMultiTaskAborter`

#### Defined in

packages/ai-tool/dist/index.d.ts:224

***

### \_generateAsyncTaskId()

> **\_generateAsyncTaskId**(`taskId`?, `aborters`?): `AsyncTaskId`

#### Parameters

• **taskId?**: `AsyncTaskId`

• **aborters?**: `TaskAbortControllers`

#### Returns

`AsyncTaskId`

#### Inherited from

`CancelableAbility._generateAsyncTaskId`

#### Defined in

packages/ai-tool/dist/index.d.ts:219

***

### $cleanMultiTaskAborter()

> **$cleanMultiTaskAborter**(`id`, `aborters`): `void`

#### Parameters

• **id**: `AsyncTaskId`

• **aborters**: `TaskAbortControllers`

#### Returns

`void`

#### Inherited from

`CancelableAbility.$cleanMultiTaskAborter`

#### Defined in

packages/ai-tool/dist/index.d.ts:222

***

### $generateAsyncTaskId()

> **$generateAsyncTaskId**(`taskId`?, `aborters`?): `undefined` \| `AsyncTaskId`

#### Parameters

• **taskId?**: `AsyncTaskId`

• **aborters?**: `TaskAbortControllers`

#### Returns

`undefined` \| `AsyncTaskId`

#### Inherited from

`CancelableAbility.$generateAsyncTaskId`

#### Defined in

packages/ai-tool/dist/index.d.ts:220

***

### abort()

> **abort**(`reason`?, `data`?): `void`

#### Parameters

• **reason?**: `string`

• **data?**: `any`

#### Returns

`void`

#### Inherited from

`CancelableAbility.abort`

#### Defined in

packages/ai-tool/dist/index.d.ts:233

***

### addModel()?

> `optional` **addModel**(`modelName`, `model`): `Promise`\<`any`\>

#### Parameters

• **modelName**: `string`

• **model**: `any`

#### Returns

`Promise`\<`any`\>

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:303

***

### arr2ObjParams()

> **arr2ObjParams**(`params`): `any`[]

#### Parameters

• **params**: `any`[]

#### Returns

`any`[]

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:83

***

### assign()

> **assign**(`src`, `options`?): `this`

Assign the values from the src object.

#### Parameters

• **src**: `any`

the source object

• **options?**: `IMergeOptions`

#### Returns

`this`

this object

#### Defined in

node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:106

***

### assignProperty()

> **assignProperty**(`src`, `name`, `value`, `attrs`?, `options`?): `void`

Assign a property of src to this object.

#### Parameters

• **src**: `any`

the src object

• **name**: `string`

the property name to assign

• **value**: `any`

the property value to assign

• **attrs?**: `any`

the attributes object

• **options?**: `IMergeOptions`

#### Returns

`void`

#### Defined in

node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:117

***

### assignPropertyTo()

> `abstract` **assignPropertyTo**(`dest`, `src`, `name`, `value`, `attrs`?, `options`?): `void`

Assign the property value from the src to destination object.

#### Parameters

• **dest**: `any`

The destination object

• **src**: `any`

The src object

• **name**: `string`

The property name

• **value**: `any`

The property value

• **attrs?**: `any`

The attributes object of the property

• **options?**: `IMergeOptions`

#### Returns

`void`

#### Defined in

node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:131

***

### assignTo()

> **assignTo**(`dest`?, `options`?): `any`

Assign this attributes to the dest object

#### Parameters

• **dest?**: `any`

the destination object

• **options?**: `IMergeOptions`

#### Returns

`any`

the dest object

#### Defined in

node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:191

***

### cleanTaskAborter()

> **cleanTaskAborter**(`aborter`): `void`

#### Parameters

• **aborter**: `TaskAbortController`

#### Returns

`void`

#### Inherited from

`CancelableAbility.cleanTaskAborter`

#### Defined in

packages/ai-tool/dist/index.d.ts:223

***

### clone()

> **clone**(`options`?): `any`

Create a new object with the same values of attributes.

#### Parameters

• **options?**: `IMergeOptions`

#### Returns

`any`

the new object

#### Defined in

node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:155

***

### cloneTo()

> **cloneTo**(`dest`, `options`?): `any`

Create and assign the values to the destination object.

#### Parameters

• **dest**: `any`

the destination object

• **options?**: `IMergeOptions`

#### Returns

`any`

the new dest object

#### Defined in

node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:148

***

### createAborter()

> **createAborter**(`params`?, `taskId`?, `raiseError`?): `TaskAbortController`

#### Parameters

• **params?**: `any`

• **taskId?**: `AsyncTaskId`

• **raiseError?**: `boolean`

#### Returns

`TaskAbortController`

#### Inherited from

`CancelableAbility.createAborter`

#### Defined in

packages/ai-tool/dist/index.d.ts:221

***

### createTaskPromise()

> **createTaskPromise**\<`Output`\>(`runTask`, `params`, `options`?): `TaskPromise`\<`Output`\>

#### Type Parameters

• **Output** = `any`

#### Parameters

• **runTask**

• **params**: `Record`\<`string`, `any`\>

• **options?**

• **options.raiseError?**: `boolean`

• **options.taskId?**: `AsyncTaskId`

#### Returns

`TaskPromise`\<`Output`\>

#### Inherited from

`CancelableAbility.createTaskPromise`

#### Defined in

packages/ai-tool/dist/index.d.ts:225

***

### defineProperties()

> `abstract` **defineProperties**(`aProperties`): `any`

Define the attributes of this object.

#### Parameters

• **aProperties**: `SimplePropDescriptors`

the defined attributes of the object

#### Returns

`any`

#### Defined in

node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:89

***

### deleteModel()?

> `optional` **deleteModel**(`modelName`): `Promise`\<`any`\>

#### Parameters

• **modelName**: `string`

#### Returns

`Promise`\<`any`\>

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:304

***

### exportTo()

> **exportTo**(`dest`, `options`?): `any`

Export attributes to the dest json object.

#### Parameters

• **dest**: `any`

the destination object

• **options?**: `IExportOptions`

#### Returns

`any`

the dest object.

#### Defined in

node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:173

***

### formatPrompt()

> **formatPrompt**(`messages`, `modelInfo`?, `options`?): `Promise`\<`undefined` \| `string`\>

#### Parameters

• **messages**: `AIChatMessageParam`[]

• **modelInfo?**: `string` \| `AIModelParams`

• **options?**

• **options.add\_generation\_prompt?**: `boolean`

• **options.chatTemplate?**: `AIPromptResult`

• **options.defaultTemplate?**: `boolean`

• **options.prompt?**: `any`

• **options.type?**: `AIPromptType`

#### Returns

`Promise`\<`undefined` \| `string`\>

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:352

***

### func()

> **func**(`__namedParameters`): `Promise`\<`any`\>

#### Parameters

• **\_\_namedParameters**

• **\_\_namedParameters.model**: `string`

• **\_\_namedParameters.options**: `AIOptions`

• **\_\_namedParameters.value**: `any`

#### Returns

`Promise`\<`any`\>

#### Defined in

[packages/ai-tool-llm-openai/src/openai-provider.ts:111](https://github.com/isdk/ai-tool-llm-openai.js/blob/3fbd49844808b871a953477e899e1386a8cded30/src/openai-provider.ts#L111)

***

### getChatTemplate()

> **getChatTemplate**(`modelInfo`?, `options`?): `Promise`\<`undefined` \| `AIPromptResult`\>

#### Parameters

• **modelInfo?**: `string` \| `AIModelParams`

• **options?**

• **options.defaultTemplate?**: `boolean`

• **options.type?**: `AIPromptType`

#### Returns

`Promise`\<`undefined` \| `AIPromptResult`\>

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:348

***

### getCurrentProvider()

> **getCurrentProvider**(): `undefined` \| `LLMProvider`

#### Returns

`undefined` \| `LLMProvider`

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:345

***

### getDefaultParameters()

> **getDefaultParameters**(`chatTemplate`, `model`): `any`

#### Parameters

• **chatTemplate**: `AIPromptSettings`

• **model**: `string`

#### Returns

`any`

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:359

***

### getFunc()

> **getFunc**(`name`?): `any`

#### Parameters

• **name?**: `string`

#### Returns

`any`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:89

***

### getFuncWithPos()

> **getFuncWithPos**(`name`?): `any`

#### Parameters

• **name?**: `string`

#### Returns

`any`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:94

***

### getModelInfo()

> **getModelInfo**(`modelName`?): `Promise`\<`undefined` \| `AIModelParams`\>

#### Parameters

• **modelName?**: `string`

#### Returns

`Promise`\<`undefined` \| `AIModelParams`\>

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:347

***

### getProperties()

> `abstract` **getProperties**(): `PropDescriptors`

Get the defined attributes.

#### Returns

`PropDescriptors`

the descriptors of properties object

#### Defined in

node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:98

***

### getRunningTask()

> **getRunningTask**(`taskId`?): `undefined` \| `TaskAbortController`

#### Parameters

• **taskId?**: `AsyncTaskId`

#### Returns

`undefined` \| `TaskAbortController`

#### Inherited from

`CancelableAbility.getRunningTask`

#### Defined in

packages/ai-tool/dist/index.d.ts:217

***

### getRunningTaskCount()

> **getRunningTaskCount**(): `number`

#### Returns

`number`

#### Inherited from

`CancelableAbility.getRunningTaskCount`

#### Defined in

packages/ai-tool/dist/index.d.ts:218

***

### hasAsyncFeature()

> **hasAsyncFeature**(`feature`): `boolean`

#### Parameters

• **feature**: `AsyncFeatureBits`

#### Returns

`boolean`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:95

***

### hasOwnProperty()

> **hasOwnProperty**(`v`): `boolean`

Determines whether an object has a property with the specified name.

#### Parameters

• **v**: `PropertyKey`

A property name.

#### Returns

`boolean`

#### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es5.d.ts:140

***

### initialize()

> **initialize**(`src`?): `this`

Initialize object and assign attribute values from src if src exists.

#### Parameters

• **src?**: `any`

#### Returns

`this`

this object.

#### Defined in

node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:139

***

### isAborted()

> **isAborted**(`taskId`?): `boolean`

#### Parameters

• **taskId?**: `AsyncTaskId`

#### Returns

`boolean`

#### Inherited from

`CancelableAbility.isAborted`

#### Defined in

packages/ai-tool/dist/index.d.ts:216

***

### isModelNameMatched()

> **isModelNameMatched**(`modelName`, `rule`?): `undefined` \| `string` \| `RegExpExecArray`

#### Parameters

• **modelName**: `string`

• **rule?**: `AIModelNameRules`

#### Returns

`undefined` \| `string` \| `RegExpExecArray`

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:336

***

### isPrototypeOf()

> **isPrototypeOf**(`v`): `boolean`

Determines whether an object exists in another object's prototype chain.

#### Parameters

• **v**: `Object`

Another object whose prototype chain is to be checked.

#### Returns

`boolean`

#### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es5.d.ts:146

***

### isSame()

> **isSame**(`src`, `options`?): `boolean`

Check the src object whether “equals” this object.

#### Parameters

• **src**: `any`

The source object

• **options?**: `IMergeOptions`

#### Returns

`boolean`

#### Defined in

node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:200

***

### isStream()

> **isStream**(`params`): `any`

#### Parameters

• **params**: `ServerFuncParams`

#### Returns

`any`

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:337

***

### listModels()?

> `optional` **listModels**(): `Promise`\<`undefined` \| `string`[]\>

#### Returns

`Promise`\<`undefined` \| `string`[]\>

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:301

***

### listProviders()

> **listProviders**(`options`?): `object`

#### Parameters

• **options?**

• **options.all?**: `boolean`

• **options.filter?**: `AIModelNameRules`

#### Returns

`object`

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:339

***

### mergeTo()

> **mergeTo**(`dest`, `options`?): `any`

Merge this attributes to dest object.

#### Parameters

• **dest**: `any`

The destination object

• **options?**: `IMergeOptions`

#### Returns

`any`

the dest object.

#### Defined in

node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:164

***

### obj2ArrParams()

> **obj2ArrParams**(`params`?): `any`[]

#### Parameters

• **params?**: `any`

#### Returns

`any`[]

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:84

***

### processModelOptions()

> **processModelOptions**(`model`, `prompt`, `options`): `Promise`\<`AIOptions`\>

#### Parameters

• **model**: `string`

• **prompt**: `any`

• **options**: `AIOptions`

#### Returns

`Promise`\<`AIOptions`\>

#### Defined in

[packages/ai-tool-llm-openai/src/openai-provider.ts:32](https://github.com/isdk/ai-tool-llm-openai.js/blob/3fbd49844808b871a953477e899e1386a8cded30/src/openai-provider.ts#L32)

***

### propertyIsEnumerable()

> **propertyIsEnumerable**(`v`): `boolean`

Determines whether a specified property is enumerable.

#### Parameters

• **v**: `PropertyKey`

A property name.

#### Returns

`boolean`

#### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es5.d.ts:152

***

### register()

> **register**(): `boolean` \| `ToolFunc`

#### Returns

`boolean` \| `ToolFunc`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:81

***

### run()

> **run**(`params`?): `Promise`\<`any`\>

#### Parameters

• **params?**: `any`

#### Returns

`Promise`\<`any`\>

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:86

***

### runAs()

> **runAs**(`name`, `params`?): `Promise`\<`any`\>

#### Parameters

• **name**: `string`

• **params?**: `any`

#### Returns

`Promise`\<`any`\>

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:87

***

### runAsSync()

> **runAsSync**(`name`, `params`?): `any`

#### Parameters

• **name**: `string`

• **params?**: `any`

#### Returns

`any`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:88

***

### runAsyncCancelableTask()

> **runAsyncCancelableTask**\<`Output`\>(`params`, `runTask`, `options`?): `TaskPromise`\<`Output`\>

#### Type Parameters

• **Output** = `any`

#### Parameters

• **params**: `undefined` \| `Record`\<`string`, `any`\>

• **runTask**

• **options?**

• **options.raiseError?**: `boolean`

• **options.taskId?**: `AsyncTaskId`

#### Returns

`TaskPromise`\<`Output`\>

#### Inherited from

`CancelableAbility.runAsyncCancelableTask`

#### Defined in

packages/ai-tool/dist/index.d.ts:229

***

### runSync()

> **runSync**(`params`?): `any`

#### Parameters

• **params?**: `any`

#### Returns

`any`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:85

***

### runWithPos()

> **runWithPos**(...`params`): `Promise`\<`any`\>

#### Parameters

• ...**params**: `any`[]

#### Returns

`Promise`\<`any`\>

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:92

***

### runWithPosAs()

> **runWithPosAs**(`name`, ...`params`): `Promise`\<`any`\>

#### Parameters

• **name**: `string`

• ...**params**: `any`[]

#### Returns

`Promise`\<`any`\>

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:93

***

### runWithPosAsSync()

> **runWithPosAsSync**(`name`, ...`params`): `any`

#### Parameters

• **name**: `string`

• ...**params**: `any`[]

#### Returns

`any`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:91

***

### runWithPosSync()

> **runWithPosSync**(...`params`): `any`

#### Parameters

• ...**params**: `any`[]

#### Returns

`any`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:90

***

### setCurrentProvider()

> **setCurrentProvider**(`name`): `void`

#### Parameters

• **name**: `string`

#### Returns

`void`

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:346

***

### toJSON()

> **toJSON**(): `any`

#### Returns

`any`

#### Defined in

node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:182

***

### toLocaleString()

> **toLocaleString**(): `string`

Returns a date converted to a string using the current locale.

#### Returns

`string`

#### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es5.d.ts:131

***

### toObject()

> **toObject**(`options`?): `any`

Convert the attributes to the json object

#### Parameters

• **options?**: `any`

#### Returns

`any`

the json object.

#### Defined in

node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/abstract.d.ts:181

***

### toString()

> **toString**(): `string`

Returns a string representation of an object.

#### Returns

`string`

#### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es5.d.ts:128

***

### unregister()

> **unregister**(): `any`

#### Returns

`any`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:82

***

### updateModel()?

> `optional` **updateModel**(`modelName`, `model`): `Promise`\<`any`\>

#### Parameters

• **modelName**: `string`

• **model**: `any`

#### Returns

`Promise`\<`any`\>

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:302

***

### valueOf()

> **valueOf**(): `Object`

Returns the primitive value of the specified object.

#### Returns

`Object`

#### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es5.d.ts:134

***

### assign()

#### assign(target, source)

> `static` **assign**\<`T`, `U`\>(`target`, `source`): `T` & `U`

Copy the values of all of the enumerable own properties from one or more source objects to a
target object. Returns the target object.

##### Type Parameters

• **T** *extends* `object`

• **U**

##### Parameters

• **target**: `T`

The target object to copy to.

• **source**: `U`

The source object from which to copy properties.

##### Returns

`T` & `U`

##### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es2015.core.d.ts:286

#### assign(target, source1, source2)

> `static` **assign**\<`T`, `U`, `V`\>(`target`, `source1`, `source2`): `T` & `U` & `V`

Copy the values of all of the enumerable own properties from one or more source objects to a
target object. Returns the target object.

##### Type Parameters

• **T** *extends* `object`

• **U**

• **V**

##### Parameters

• **target**: `T`

The target object to copy to.

• **source1**: `U`

The first source object from which to copy properties.

• **source2**: `V`

The second source object from which to copy properties.

##### Returns

`T` & `U` & `V`

##### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es2015.core.d.ts:295

#### assign(target, source1, source2, source3)

> `static` **assign**\<`T`, `U`, `V`, `W`\>(`target`, `source1`, `source2`, `source3`): `T` & `U` & `V` & `W`

Copy the values of all of the enumerable own properties from one or more source objects to a
target object. Returns the target object.

##### Type Parameters

• **T** *extends* `object`

• **U**

• **V**

• **W**

##### Parameters

• **target**: `T`

The target object to copy to.

• **source1**: `U`

The first source object from which to copy properties.

• **source2**: `V`

The second source object from which to copy properties.

• **source3**: `W`

The third source object from which to copy properties.

##### Returns

`T` & `U` & `V` & `W`

##### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es2015.core.d.ts:305

#### assign(target, sources)

> `static` **assign**(`target`, ...`sources`): `any`

Copy the values of all of the enumerable own properties from one or more source objects to a
target object. Returns the target object.

##### Parameters

• **target**: `object`

The target object to copy to.

• ...**sources**: `any`[]

One or more source objects from which to copy properties

##### Returns

`any`

##### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es2015.core.d.ts:313

***

### create()

#### create(o)

> `static` **create**(`o`): `any`

Creates an object that has the specified prototype or that has null prototype.

##### Parameters

• **o**: `null` \| `object`

Object to use as a prototype. May be null.

##### Returns

`any`

##### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es5.d.ts:188

#### create(o, properties)

> `static` **create**(`o`, `properties`): `any`

Creates an object that has the specified prototype, and that optionally contains specified properties.

##### Parameters

• **o**: `null` \| `object`

Object to use as a prototype. May be null

• **properties**: `PropertyDescriptorMap` & `ThisType`\<`any`\>

JavaScript object that contains one or more property descriptors.

##### Returns

`any`

##### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es5.d.ts:195

***

### defineProperties()

> `static` **defineProperties**(`aTarget`, `aProperties`, `recreate`?): `any`

Adds one or more properties to an object, and/or modifies attributes of existing properties.

#### Parameters

• **aTarget**: `any`

• **aProperties**: `PropDescriptors`

• **recreate?**: `boolean`

#### Returns

`any`

#### Defined in

node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/advance.d.ts:11

***

### defineProperty()

> `static` **defineProperty**\<`T`\>(`o`, `p`, `attributes`): `T`

Adds a property to an object, or modifies attributes of an existing property.

#### Type Parameters

• **T**

#### Parameters

• **o**: `T`

Object on which to add or modify the property. This can be a native JavaScript object (that is, a user-defined object or a built in object) or a DOM object.

• **p**: `PropertyKey`

The property name.

• **attributes**: `PropertyDescriptor` & `ThisType`\<`any`\>

Descriptor for the property. It can be for a data property or an accessor property.

#### Returns

`T`

#### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es5.d.ts:203

***

### entries()

#### entries(o)

> `static` **entries**\<`T`\>(`o`): [`string`, `T`][]

Returns an array of key/values of the enumerable own properties of an object

##### Type Parameters

• **T**

##### Parameters

• **o**: `object` \| `ArrayLike`\<`T`\>

Object that contains the properties and methods. This can be an object that you created or an existing Document Object Model (DOM) object.

##### Returns

[`string`, `T`][]

##### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es2017.object.d.ts:36

#### entries(o)

> `static` **entries**(`o`): [`string`, `any`][]

Returns an array of key/values of the enumerable own properties of an object

##### Parameters

• **o**

Object that contains the properties and methods. This can be an object that you created or an existing Document Object Model (DOM) object.

##### Returns

[`string`, `any`][]

##### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es2017.object.d.ts:42

***

### freeze()

#### freeze(f)

> `static` **freeze**\<`T`\>(`f`): `T`

Prevents the modification of existing property attributes and values, and prevents the addition of new properties.

##### Type Parameters

• **T** *extends* `Function`

##### Parameters

• **f**: `T`

Object on which to lock the attributes.

##### Returns

`T`

##### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es5.d.ts:222

#### freeze(o)

> `static` **freeze**\<`T`, `U`\>(`o`): `Readonly`\<`T`\>

Prevents the modification of existing property attributes and values, and prevents the addition of new properties.

##### Type Parameters

• **T** *extends* `object`

• **U** *extends* `string` \| `number` \| `bigint` \| `boolean` \| `symbol`

##### Parameters

• **o**: `T`

Object on which to lock the attributes.

##### Returns

`Readonly`\<`T`\>

##### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es5.d.ts:228

#### freeze(o)

> `static` **freeze**\<`T`\>(`o`): `Readonly`\<`T`\>

Prevents the modification of existing property attributes and values, and prevents the addition of new properties.

##### Type Parameters

• **T**

##### Parameters

• **o**: `T`

Object on which to lock the attributes.

##### Returns

`Readonly`\<`T`\>

##### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es5.d.ts:234

***

### fromEntries()

#### fromEntries(entries)

> `static` **fromEntries**\<`T`\>(`entries`): `object`

Returns an object created by key-value entries for properties and methods

##### Type Parameters

• **T** = `any`

##### Parameters

• **entries**: `Iterable`\<readonly [`PropertyKey`, `T`], `any`, `any`\>

An iterable object that contains key-value entries for properties and methods.

##### Returns

`object`

##### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es2019.object.d.ts:26

#### fromEntries(entries)

> `static` **fromEntries**(`entries`): `any`

Returns an object created by key-value entries for properties and methods

##### Parameters

• **entries**: `Iterable`\<readonly `any`[], `any`, `any`\>

An iterable object that contains key-value entries for properties and methods.

##### Returns

`any`

##### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es2019.object.d.ts:32

***

### get()

> `static` **get**(`name`): `ToolFunc`

#### Parameters

• **name**: `string`

#### Returns

`ToolFunc`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:65

***

### getAllByTag()

> `static` **getAllByTag**(`tagName`): `ToolFunc`[]

#### Parameters

• **tagName**: `string`

#### Returns

`ToolFunc`[]

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:68

***

### getByModel()

> `static` **getByModel**(`modelName`?): `undefined` \| `LLMProvider`

#### Parameters

• **modelName?**: `string`

#### Returns

`undefined` \| `LLMProvider`

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:334

***

### getByTag()

> `static` **getByTag**(`tagName`): `undefined` \| `ToolFunc`

#### Parameters

• **tagName**: `string`

#### Returns

`undefined` \| `ToolFunc`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:67

***

### getCurrentProvider()

> `static` **getCurrentProvider**(): `undefined` \| `LLMProvider`

#### Returns

`undefined` \| `LLMProvider`

#### Defined in

packages/ai-tool-llm/dist/index.d.ts:335

***

### getFunc()

> `static` **getFunc**(`name`): `any`

#### Parameters

• **name**: `string`

#### Returns

`any`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:72

***

### getFuncWithPos()

> `static` **getFuncWithPos**(`name`): `any`

#### Parameters

• **name**: `string`

#### Returns

`any`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:75

***

### getOwnPropertyDescriptor()

> `static` **getOwnPropertyDescriptor**(`o`, `p`): `undefined` \| `PropertyDescriptor`

Gets the own property descriptor of the specified object.
An own property descriptor is one that is defined directly on the object and is not inherited from the object's prototype.

#### Parameters

• **o**: `any`

Object that contains the property.

• **p**: `PropertyKey`

Name of the property.

#### Returns

`undefined` \| `PropertyDescriptor`

#### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es5.d.ts:175

***

### getOwnPropertyDescriptors()

> `static` **getOwnPropertyDescriptors**\<`T`\>(`o`): \{ \[P in string \| number \| symbol\]: TypedPropertyDescriptor\<T\[P\]\> \} & `object`

Returns an object containing all own property descriptors of an object

#### Type Parameters

• **T**

#### Parameters

• **o**: `T`

Object that contains the properties and methods. This can be an object that you created or an existing Document Object Model (DOM) object.

#### Returns

\{ \[P in string \| number \| symbol\]: TypedPropertyDescriptor\<T\[P\]\> \} & `object`

#### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es2017.object.d.ts:48

***

### getOwnPropertyNames()

> `static` **getOwnPropertyNames**(`o`): `string`[]

Returns the names of the own properties of an object. The own properties of an object are those that are defined directly
on that object, and are not inherited from the object's prototype. The properties of an object include both fields (objects) and functions.

#### Parameters

• **o**: `any`

Object that contains the own properties.

#### Returns

`string`[]

#### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es5.d.ts:182

***

### getOwnPropertySymbols()

> `static` **getOwnPropertySymbols**(`o`): `symbol`[]

Returns an array of all symbol properties found directly on object o.

#### Parameters

• **o**: `any`

Object to retrieve the symbols from.

#### Returns

`symbol`[]

#### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es2015.core.d.ts:319

***

### getProperties()

> `static` **getProperties**(): `PropDescriptors`

get all properties descriptor include inherited.

#### Returns

`PropDescriptors`

#### Defined in

node\_modules/.pnpm/property-manager@2.0.0/node\_modules/property-manager/lib/advance.d.ts:10

***

### getPrototypeOf()

> `static` **getPrototypeOf**(`o`): `any`

Returns the prototype of an object.

#### Parameters

• **o**: `any`

The object that references the prototype.

#### Returns

`any`

#### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es5.d.ts:167

***

### hasAsyncFeature()

> `static` **hasAsyncFeature**(`feature`): `boolean`

#### Parameters

• **feature**: `AsyncFeatureBits`

#### Returns

`boolean`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:69

***

### is()

> `static` **is**(`value1`, `value2`): `boolean`

Returns true if the values are the same value, false otherwise.

#### Parameters

• **value1**: `any`

The first value.

• **value2**: `any`

The second value.

#### Returns

`boolean`

#### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es2015.core.d.ts:332

***

### isExtensible()

> `static` **isExtensible**(`o`): `boolean`

Returns a value that indicates whether new properties can be added to an object.

#### Parameters

• **o**: `any`

Object to test.

#### Returns

`boolean`

#### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es5.d.ts:258

***

### isFrozen()

> `static` **isFrozen**(`o`): `boolean`

Returns true if existing property attributes and values cannot be modified in an object, and new properties cannot be added to the object.

#### Parameters

• **o**: `any`

Object to test.

#### Returns

`boolean`

#### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es5.d.ts:252

***

### isSealed()

> `static` **isSealed**(`o`): `boolean`

Returns true if existing property attributes cannot be modified in an object and new properties cannot be added to the object.

#### Parameters

• **o**: `any`

Object to test.

#### Returns

`boolean`

#### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es5.d.ts:246

***

### keys()

#### keys(o)

> `static` **keys**(`o`): `string`[]

Returns the names of the enumerable string properties and methods of an object.

##### Parameters

• **o**: `object`

Object that contains the properties and methods. This can be an object that you created or an existing Document Object Model (DOM) object.

##### Returns

`string`[]

##### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es5.d.ts:264

#### keys(o)

> `static` **keys**(`o`): `string`[]

Returns the names of the enumerable string properties and methods of an object.

##### Parameters

• **o**

Object that contains the properties and methods. This can be an object that you created or an existing Document Object Model (DOM) object.

##### Returns

`string`[]

##### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es2015.core.d.ts:325

***

### list()

> `static` **list**(): `Funcs`

#### Returns

`Funcs`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:66

***

### preventExtensions()

> `static` **preventExtensions**\<`T`\>(`o`): `T`

Prevents the addition of new properties to an object.

#### Type Parameters

• **T**

#### Parameters

• **o**: `T`

Object to make non-extensible.

#### Returns

`T`

#### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es5.d.ts:240

***

### register()

#### register(name, options)

> `static` **register**(`name`, `options`): `boolean` \| `ToolFunc`

##### Parameters

• **name**: `string`

• **options**: `FuncItem`

##### Returns

`boolean` \| `ToolFunc`

##### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:76

#### register(func, options)

> `static` **register**(`func`, `options`): `boolean` \| `ToolFunc`

##### Parameters

• **func**: `Function`

• **options**: `FuncItem`

##### Returns

`boolean` \| `ToolFunc`

##### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:77

#### register(name, options)

> `static` **register**(`name`, `options`?): `boolean` \| `ToolFunc`

##### Parameters

• **name**: `string` \| `Function` \| `ToolFunc` \| `FuncItem`

• **options?**: `FuncItem`

##### Returns

`boolean` \| `ToolFunc`

##### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:78

***

### run()

> `static` **run**(`name`, `params`?): `Promise`\<`any`\>

#### Parameters

• **name**: `string`

• **params?**: `any`

#### Returns

`Promise`\<`any`\>

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:70

***

### runSync()

> `static` **runSync**(`name`, `params`?): `any`

#### Parameters

• **name**: `string`

• **params?**: `any`

#### Returns

`any`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:71

***

### runWithPos()

> `static` **runWithPos**(`name`, ...`params`): `Promise`\<`any`\>

#### Parameters

• **name**: `string`

• ...**params**: `any`[]

#### Returns

`Promise`\<`any`\>

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:73

***

### runWithPosSync()

> `static` **runWithPosSync**(`name`, ...`params`): `any`

#### Parameters

• **name**: `string`

• ...**params**: `any`[]

#### Returns

`any`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:74

***

### seal()

> `static` **seal**\<`T`\>(`o`): `T`

Prevents the modification of attributes of existing properties, and prevents the addition of new properties.

#### Type Parameters

• **T**

#### Parameters

• **o**: `T`

Object on which to lock the attributes.

#### Returns

`T`

#### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es5.d.ts:216

***

### setPrototypeOf()

> `static` **setPrototypeOf**(`o`, `proto`): `any`

Sets the prototype of a specified object o to object proto or null. Returns the object o.

#### Parameters

• **o**: `any`

The object to change its prototype.

• **proto**: `null` \| `object`

The value of the new prototype or null.

#### Returns

`any`

#### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es2015.core.d.ts:339

***

### unregister()

> `static` **unregister**(`name`): `undefined` \| `ToolFunc`

#### Parameters

• **name**: `string`

#### Returns

`undefined` \| `ToolFunc`

#### Defined in

packages/ai-tool/dist/index-DUEs7T8M.d.ts:79

***

### values()

#### values(o)

> `static` **values**\<`T`\>(`o`): `T`[]

Returns an array of values of the enumerable own properties of an object

##### Type Parameters

• **T**

##### Parameters

• **o**: `object` \| `ArrayLike`\<`T`\>

Object that contains the properties and methods. This can be an object that you created or an existing Document Object Model (DOM) object.

##### Returns

`T`[]

##### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es2017.object.d.ts:24

#### values(o)

> `static` **values**(`o`): `any`[]

Returns an array of values of the enumerable own properties of an object

##### Parameters

• **o**

Object that contains the properties and methods. This can be an object that you created or an existing Document Object Model (DOM) object.

##### Returns

`any`[]

##### Defined in

node\_modules/.pnpm/typescript@5.6.2/node\_modules/typescript/lib/lib.es2017.object.d.ts:30
