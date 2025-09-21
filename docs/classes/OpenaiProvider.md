[**@isdk/ai-tool-llm-openai**](../README.md)

***

[@isdk/ai-tool-llm-openai](../globals.md) / OpenaiProvider

# Class: OpenaiProvider

Defined in: [@isdk/ai-tools/packages/ai-tool-llm-openai/src/openai-provider.ts:20](https://github.com/isdk/ai-tool-llm-openai.js/blob/daf11bb0e357dbe1c5084a6514955d7a05201a3b/src/openai-provider.ts#L20)

## Extends

- `CancelableAbility`.`LLMProvider`

## Indexable

\[`name`: `string`\]: `any`

## Constructors

### Constructor

> **new OpenaiProvider**(`name`, `options?`): `OpenaiProvider`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:413

Initializes a new `ToolFunc` instance.

#### Parameters

##### name

Can be a function name, a function implementation, or a configuration object.

`string` | `Function` | `FuncItem`

##### options?

`any`

Configuration options if not provided in the first argument.

#### Returns

`OpenaiProvider`

#### Inherited from

`CancelableAbility.constructor`

## Properties

### \_\_task\_aborter

> **\_\_task\_aborter**: `undefined` \| `TaskAbortController` \| `TaskAbortControllers`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:1074

#### Inherited from

`CancelableAbility.__task_aborter`

***

### \_\_task\_semaphore

> **\_\_task\_semaphore**: `undefined` \| `Semaphore`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:1075

#### Inherited from

`CancelableAbility.__task_semaphore`

***

### \_asyncFeatures?

> `optional` **\_asyncFeatures**: `number`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:1066

#### Inherited from

`CancelableAbility._asyncFeatures`

***

### \_isReadyFn?

> `optional` **\_isReadyFn**: `SemaphoreIsReadyFuncType`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:1068

#### Inherited from

`CancelableAbility._isReadyFn`

***

### \_maxTaskConcurrency

> **\_maxTaskConcurrency**: `undefined` \| `number`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:1067

#### Inherited from

`CancelableAbility._maxTaskConcurrency`

***

### $attributes

> **$attributes**: `Properties`

Defined in: property-manager.js/lib/advance.d.ts:5

***

### alias?

> `optional` **alias**: `string` \| `string`[]

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:124

Optional aliases for the function name.

***

### apiKey?

> `optional` **apiKey**: `string`

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:153

***

### apiUrl?

> `optional` **apiUrl**: `string`

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:154

***

### asyncFeatures?

> `optional` **asyncFeatures**: `number`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:138

A bitmask representing asynchronous features supported by the function, built from `AsyncFeatureBits`.
This allows the system to understand if a function supports capabilities like cancellation or multi-tasking.

#### See

AsyncFeatureBits from `@src/utils/cancelable-ability.ts`

#### Example

```ts
import { AsyncFeatures } from './utils';
const func = new ToolFunc({
  name: 'cancellableTask',
  asyncFeatures: AsyncFeatures.Cancelable | AsyncFeatures.MultiTask,
  // ...
});
```

***

### cleanMultiTaskAborter()

> **cleanMultiTaskAborter**: (`id`, `aborters`) => `void`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:1073

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

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:125

The initial value of Object.prototype.constructor is the standard built-in Object constructor.

***

### defaultOptions

> **defaultOptions**: `object`

Defined in: property-manager.js/lib/abstract.d.ts:74

The default options for export and assign

#### assign?

> `optional` **assign**: `IMergeOptions`

#### export?

> `optional` **export**: `IMergeOptions`

***

### depends

> **depends**: `object`

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:336

A map of dependencies this function has on other tool functions.
Declaring dependencies ensures that they are automatically registered when this function is registered.
This is crucial for building modular functions that rely on each other without needing to manage registration order manually.

#### Index Signature

\[`name`: `string`\]: `ToolFunc`

#### Example

```ts
const helperFunc = new ToolFunc({ name: 'helper', func: () => 'world' });
const mainFunc = new ToolFunc({
  name: 'main',
  depends: {
    helper: helperFunc,
  },
  func() {
    // We can now safely run the dependency
    const result = this.runSync('helper');
    return `Hello, ${result}`;
  }
});
// When mainFunc is registered, helperFunc will be registered automatically.
mainFunc.register();
```

***

### description

> **description**: `string`

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:317

A detailed description of what the function does.

***

### generateAsyncTaskId()

> **generateAsyncTaskId**: (`taskId?`, `aborters?`) => `AsyncTaskId`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:1072

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

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:113

If true, indicates that this function should be treated as a server-side API.

***

### model?

> `optional` **model**: `string`

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:310

***

### name?

> `optional` **name**: `string`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:70

The unique name of the function.

***

### nonExported1stChar

> **nonExported1stChar**: `string`

Defined in: property-manager.js/lib/abstract.d.ts:78

the property with the default prefix '$' will not be exported.

***

### params

> **params**: `object`

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:318

Parameter definitions, which can be an object mapping names to definitions or an array for positional parameters.

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

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:151

***

### result

> **result**: `string`

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:335

The expected return type of the function, described as a string or a JSON schema object.

***

### rule

> **rule**: (`string` \| `RegExp`)[]

Defined in: [@isdk/ai-tools/packages/ai-tool-llm-openai/src/openai-provider.ts:22](https://github.com/isdk/ai-tool-llm-openai.js/blob/daf11bb0e357dbe1c5084a6514955d7a05201a3b/src/openai-provider.ts#L22)

***

### scope?

> `optional` **scope**: `any`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:85

The execution scope or context (`this`) for the function.

***

### setup()?

> `optional` **setup**: (`this`, `options?`) => `void`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:108

A lifecycle hook called once during the `ToolFunc` instance's initialization.
It allows for initial setup, state configuration, or property modification on the instance
before it is used or registered. The `this` context is the `ToolFunc` instance itself.

#### Parameters

##### this

`ToolFunc`

##### options?

`FuncItem`

The configuration options for the function.

#### Returns

`void`

#### Example

```ts
const myFunc = new ToolFunc({
  name: 'myFunc',
  customState: 'initial',
  setup() {
    // `this` is the myFunc instance
    this.customState = 'configured';
  }
});
console.log(myFunc.customState); // Outputs: 'configured'
```

***

### stream?

> `optional` **stream**: `boolean`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:119

If true, indicates that the function has the *capability* to stream its output.
Whether a specific call is streamed is determined by a `stream` property in the runtime parameters.

***

### supports

> **supports**: `AIModelType` \| `AIModelType`[]

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:152

***

### tags?

> `optional` **tags**: `string` \| `string`[]

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:90

Tags for grouping or filtering functions.

***

### title?

> `optional` **title**: `string`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:173

A concise, human-readable title for the function, often used in UI or by AI.

***

### aliases

> `static` **aliases**: `object`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:295

A static map of aliases to their corresponding function names.

#### Index Signature

\[`name`: `string`\]: `string`

***

### current?

> `static` `optional` **current**: `string`

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:313

***

### dataPath

> `static` **dataPath**: `string`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:304

A conventional property to designate a file path for saving the registered `ToolFunc` data.
Note: The `ToolFunc` class itself does not implement persistence logic. It is up to the
developer to use this path to save and load the `ToolFunc.items` registry if needed.

***

### items

> `static` **items**: `object`

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:314

A static registry of all `ToolFunc` instances, indexed by name.

#### Index Signature

\[`name`: `string`\]: `LLMProvider`

## Accessors

### maxTaskConcurrency

#### Get Signature

> **get** **maxTaskConcurrency**(): `undefined` \| `number`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:1076

##### Returns

`undefined` \| `number`

#### Inherited from

`CancelableAbility.maxTaskConcurrency`

***

### semaphore

#### Get Signature

> **get** **semaphore**(): `undefined` \| `Semaphore`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:1077

##### Returns

`undefined` \| `Semaphore`

#### Inherited from

`CancelableAbility.semaphore`

## Methods

### \_cleanMultiTaskAborter()

> **\_cleanMultiTaskAborter**(`id`, `aborters`): `void`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:1089

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

> **\_generateAsyncTaskId**(`taskId?`, `aborters?`): `AsyncTaskId`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:1084

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

### \_getChatTemplate()

> **\_getChatTemplate**(`modelInfo?`, `options?`): `Promise`\<`undefined` \| `AIPromptResult`\>

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:356

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

### $cleanMultiTaskAborter()

> **$cleanMultiTaskAborter**(`id`, `aborters`): `void`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:1087

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

> **$generateAsyncTaskId**(`taskId?`, `aborters?`): `undefined` \| `AsyncTaskId`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:1085

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

> **abort**(`reason?`, `data?`): `void`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:1099

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

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:308

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

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:431

Converts an array of positional arguments into a named parameters object.
This is used internally to support functions defined with named parameters.

#### Parameters

##### params

`any`[]

An array of positional arguments.

#### Returns

`any`[]

An array containing a single parameters object.

***

### assign()

> **assign**(`src`, `options?`): `this`

Defined in: property-manager.js/lib/abstract.d.ts:106

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

> **assignProperty**(`src`, `name`, `value`, `attrs?`, `options?`): `void`

Defined in: property-manager.js/lib/abstract.d.ts:117

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

> `abstract` **assignPropertyTo**(`dest`, `src`, `name`, `value`, `attrs?`, `options?`): `void`

Defined in: property-manager.js/lib/abstract.d.ts:131

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

> **assignTo**(`dest?`, `options?`): `any`

Defined in: property-manager.js/lib/abstract.d.ts:191

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

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:1088

#### Parameters

##### aborter

`TaskAbortController`

#### Returns

`void`

#### Inherited from

`CancelableAbility.cleanTaskAborter`

***

### clone()

> **clone**(`options?`): `any`

Defined in: property-manager.js/lib/abstract.d.ts:155

Create a new object with the same values of attributes.

#### Parameters

##### options?

`IMergeOptions`

#### Returns

`any`

the new object

***

### cloneTo()

> **cloneTo**(`dest`, `options?`): `any`

Defined in: property-manager.js/lib/abstract.d.ts:148

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

> **countTokens**(`text`, `options?`): `Promise`\<`number`\>

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:377

#### Parameters

##### text

`string` | `AIChatMessageParam`[]

##### options?

`AITokenizeOptions`

#### Returns

`Promise`\<`number`\>

***

### createAborter()

> **createAborter**(`params?`, `taskId?`, `raiseError?`): `TaskAbortController`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:1086

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

> **createTaskPromise**\<`Output`\>(`runTask`, `params`, `options?`): `TaskPromise`\<`Output`\>

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:1090

#### Type Parameters

##### Output

`Output` = `any`

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

Defined in: property-manager.js/lib/abstract.d.ts:89

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

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:309

#### Parameters

##### modelName

`string`

#### Returns

`Promise`\<`any`\>

***

### exportTo()

> **exportTo**(`dest`, `options?`): `any`

Defined in: property-manager.js/lib/abstract.d.ts:173

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

> **formatPrompt**(`messages`, `modelInfo?`, `options?`): `Promise`\<`undefined` \| `string`\>

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:367

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

Defined in: [@isdk/ai-tools/packages/ai-tool-llm-openai/src/openai-provider.ts:111](https://github.com/isdk/ai-tool-llm-openai.js/blob/daf11bb0e357dbe1c5084a6514955d7a05201a3b/src/openai-provider.ts#L111)

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

> **getChatTemplate**(`chatTemplate?`, `options?`): `Promise`\<`AIPromptResult`\>

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:361

#### Parameters

##### chatTemplate?

`string` | `AIPromptResult`

##### options?

###### defaultTemplate?

`boolean`

###### modelInfo?

`string` \| `AIModelParams`

###### provider?

`string`

###### type?

`AIPromptType`

#### Returns

`Promise`\<`AIPromptResult`\>

***

### getCurrentProvider()

> **getCurrentProvider**(): `undefined` \| `LLMProvider`

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:353

#### Returns

`undefined` \| `LLMProvider`

***

### getDefaultParameters()

> **getDefaultParameters**(`chatTemplate`, `model`): `any`

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:375

#### Parameters

##### chatTemplate

`AIPromptSettings`

##### model

`string`

#### Returns

`any`

***

### getFunc()

> **getFunc**(`name?`): `any`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:475

Gets a bound function reference for execution with named parameters.
If a name is provided, it retrieves a different function from the registry.
Otherwise, it returns a bound version of this instance's `runSync`.

#### Parameters

##### name?

`string`

Optional name of the function to retrieve.

#### Returns

`any`

A function reference or `undefined` if not found.

***

### getFuncWithPos()

> **getFuncWithPos**(`name?`): `any`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:513

Gets a bound function reference suitable for positional argument execution.
If a name is provided, it retrieves a different function from the registry.
Otherwise, it returns a bound version of this instance's `runWithPosSync`.

#### Parameters

##### name?

`string`

Optional name of the function to retrieve.

#### Returns

`any`

A function reference or `undefined` if not found.

***

### getModelInfo()

> **getModelInfo**(`modelName?`, `options?`): `Promise`\<`undefined` \| `AIModelParams`\>

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:355

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

Defined in: property-manager.js/lib/abstract.d.ts:98

Get the defined attributes.

#### Returns

`PropDescriptors`

the descriptors of properties object

***

### getProvider()

> **getProvider**(`name`): `ToolFunc`

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:352

#### Parameters

##### name

`string`

#### Returns

`ToolFunc`

***

### getRunningTask()

> **getRunningTask**(`taskId?`): `undefined` \| `TaskAbortController`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:1082

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

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:1083

#### Returns

`number`

#### Inherited from

`CancelableAbility.getRunningTaskCount`

***

### getSemaphore()

> **getSemaphore**(`isReadyFn?`): `undefined` \| `Semaphore`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:1078

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

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:519

Checks if the current function instance supports a specific async feature.

#### Parameters

##### feature

`AsyncFeatureBits`

The async feature bit to check for.

#### Returns

`boolean`

`true` if the feature is supported, otherwise `false`.

#### Inherited from

`CancelableAbility.hasAsyncFeature`

***

### hasOwnProperty()

> **hasOwnProperty**(`v`): `boolean`

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:140

Determines whether an object has a property with the specified name.

#### Parameters

##### v

`PropertyKey`

A property name.

#### Returns

`boolean`

***

### initialize()

> **initialize**(`src?`): `this`

Defined in: property-manager.js/lib/abstract.d.ts:139

Initialize object and assign attribute values from src if src exists.

#### Parameters

##### src?

`any`

#### Returns

`this`

this object.

***

### isAborted()

> **isAborted**(`taskId?`): `boolean`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:1081

#### Parameters

##### taskId?

`AsyncTaskId`

#### Returns

`boolean`

#### Inherited from

`CancelableAbility.isAborted`

***

### isModelNameMatched()

> **isModelNameMatched**(`modelName`, `rule?`): `undefined` \| `string` \| `RegExpExecArray`

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:343

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

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:146

Determines whether an object exists in another object's prototype chain.

#### Parameters

##### v

`Object`

Another object whose prototype chain is to be checked.

#### Returns

`boolean`

***

### isSame()

> **isSame**(`src`, `options?`): `boolean`

Defined in: property-manager.js/lib/abstract.d.ts:200

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

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:344

Determines if a function call should produce a stream.

The logic is as follows:
1. It first checks if the function is generally capable of streaming (`this.stream`).
2. If it is, it then checks if a `stream` parameter is formally declared in the function's `params` definition.
3. If both are true, the method returns the value of the `stream` property from the runtime `params` object.
Otherwise, it returns the function's static `stream` capability.

#### Parameters

##### params

`ServerFuncParams`

The runtime parameters passed to the function call.

#### Returns

`any`

`true` if the call should be streamed, `false` or `undefined` otherwise.

***

### listModels()?

> `optional` **listModels**(): `Promise`\<`undefined` \| `string`[]\>

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:306

#### Returns

`Promise`\<`undefined` \| `string`[]\>

***

### listProviders()

> **listProviders**(`options?`): `object`

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:346

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

> **mergeTo**(`dest`, `options?`): `any`

Defined in: property-manager.js/lib/abstract.d.ts:164

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

> **obj2ArrParams**(`params?`): `any`[]

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:438

Converts a named parameters object into an array of positional arguments.
This is used for functions defined with positional parameters.

#### Parameters

##### params?

`any`

A named parameters object.

#### Returns

`any`[]

An array of positional arguments.

***

### processModelOptions()

> **processModelOptions**(`model`, `prompt`, `options`): `Promise`\<`AIOptions`\>

Defined in: [@isdk/ai-tools/packages/ai-tool-llm-openai/src/openai-provider.ts:32](https://github.com/isdk/ai-tool-llm-openai.js/blob/daf11bb0e357dbe1c5084a6514955d7a05201a3b/src/openai-provider.ts#L32)

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

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:152

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

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:419

Registers the current `ToolFunc` instance into the static registry.
Also registers any declared dependencies.

#### Returns

`boolean` \| `ToolFunc`

The instance itself upon successful registration, or `false` if it already exists.

***

### run()

> **run**(`params?`): `Promise`\<`any`\>

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:451

Executes the function asynchronously with a named parameters object.

#### Parameters

##### params?

`any`

The parameters object for the function.

#### Returns

`Promise`\<`any`\>

A promise that resolves with the function's result.

***

### runAs()

> **runAs**(`name`, `params?`): `Promise`\<`any`\>

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:459

Asynchronously executes another registered function by name.
This method delegates to `runAsSync()` internally.

#### Parameters

##### name

`string`

The name of the target function to run.

##### params?

`any`

Optional parameters to pass to the function.

#### Returns

`Promise`\<`any`\>

A promise that resolves with the result of the function execution.

***

### runAsSync()

> **runAsSync**(`name`, `params?`): `any`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:467

Synchronously executes another registered function by name.
This is a convenience method that forwards the call to the static `runSync()` method.

#### Parameters

##### name

`string`

The name of the target function to run.

##### params?

`any`

Optional parameters to pass to the function.

#### Returns

`any`

The result of the function execution.

***

### runAsyncCancelableTask()

> **runAsyncCancelableTask**\<`Output`\>(`params`, `runTask`, `options?`): `TaskPromise`\<`Output`\>

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:1094

#### Type Parameters

##### Output

`Output` = `any`

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

> **runSync**(`params?`): `any`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:445

Executes the function synchronously with a named parameters object.

#### Parameters

##### params?

`any`

The parameters object for the function.

#### Returns

`any`

The result of the function execution.

#### Throws

Will throw an error if an array of parameters is passed to a function that expects an object.

***

### runWithPos()

> **runWithPos**(...`params`): `Promise`\<`any`\>

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:497

Executes the function asynchronously using positional arguments.
Delegates to `runWithPosSync()` internally.

#### Parameters

##### params

...`any`[]

Positional arguments passed to the function.

#### Returns

`Promise`\<`any`\>

A promise that resolves with the result of the function execution.

***

### runWithPosAs()

> **runWithPosAs**(`name`, ...`params`): `Promise`\<`any`\>

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:505

Asynchronously executes another function by name using positional arguments.
Delegates to `runWithPosAsSync()` internally.

#### Parameters

##### name

`string`

The name of the target function to run.

##### params

...`any`[]

Positional arguments to pass to the function.

#### Returns

`Promise`\<`any`\>

A promise that resolves with the result of the function execution.

***

### runWithPosAsSync()

> **runWithPosAsSync**(`name`, ...`params`): `any`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:490

Synchronously executes another function by name using positional arguments.
This is a convenience wrapper around the static `runWithPosSync()` method.

#### Parameters

##### name

`string`

The name of the target function to run.

##### params

...`any`[]

Positional arguments to pass to the function.

#### Returns

`any`

The result of the function execution.

***

### runWithPosSync()

> **runWithPosSync**(...`params`): `any`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:482

Executes the function synchronously using positional arguments.
If the function expects named parameters, it converts the arguments automatically.

#### Parameters

##### params

...`any`[]

Positional arguments passed to the function.

#### Returns

`any`

The result of the function execution.

***

### setCurrentProvider()

> **setCurrentProvider**(`name`): `void`

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:354

#### Parameters

##### name

`string`

#### Returns

`void`

***

### toJSON()

> **toJSON**(): `any`

Defined in: property-manager.js/lib/abstract.d.ts:182

#### Returns

`any`

***

### tokenize()

> **tokenize**(`text`, `options?`): `Promise`\<`number`[]\>

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:376

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

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:131

Returns a date converted to a string using the current locale.

#### Returns

`string`

***

### toObject()

> **toObject**(`options?`): `any`

Defined in: property-manager.js/lib/abstract.d.ts:181

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

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:128

Returns a string representation of an object.

#### Returns

`string`

***

### unregister()

> **unregister**(): `any`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:424

Removes the current `ToolFunc` instance from the static registry.

#### Returns

`any`

The instance that was unregistered.

***

### updateModel()?

> `optional` **updateModel**(`modelName`, `model`): `Promise`\<`any`\>

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:307

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

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:134

Returns the primitive value of the specified object.

#### Returns

`Object`

***

### assign()

#### Call Signature

> `static` **assign**\<`T`, `U`\>(`target`, `source`): `T` & `U`

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2015.core.d.ts:286

Copy the values of all of the enumerable own properties from one or more source objects to a
target object. Returns the target object.

##### Type Parameters

###### T

`T` *extends* `object`

###### U

`U`

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

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2015.core.d.ts:295

Copy the values of all of the enumerable own properties from one or more source objects to a
target object. Returns the target object.

##### Type Parameters

###### T

`T` *extends* `object`

###### U

`U`

###### V

`V`

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

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2015.core.d.ts:305

Copy the values of all of the enumerable own properties from one or more source objects to a
target object. Returns the target object.

##### Type Parameters

###### T

`T` *extends* `object`

###### U

`U`

###### V

`V`

###### W

`W`

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

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2015.core.d.ts:313

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

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:188

Creates an object that has the specified prototype or that has null prototype.

##### Parameters

###### o

Object to use as a prototype. May be null.

`null` | `object`

##### Returns

`any`

#### Call Signature

> `static` **create**(`o`, `properties`): `any`

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:195

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

> `static` **defineProperties**(`aTarget`, `aProperties`, `recreate?`): `any`

Defined in: property-manager.js/lib/advance.d.ts:11

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

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:203

Adds a property to an object, or modifies attributes of an existing property.

#### Type Parameters

##### T

`T`

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

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2017.object.d.ts:36

Returns an array of key/values of the enumerable own properties of an object

##### Type Parameters

###### T

`T`

##### Parameters

###### o

Object that contains the properties and methods. This can be an object that you created or an existing Document Object Model (DOM) object.

\{\[`s`: `string`\]: `T`; \} | `ArrayLike`\<`T`\>

##### Returns

\[`string`, `T`\][]

#### Call Signature

> `static` **entries**(`o`): \[`string`, `any`\][]

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2017.object.d.ts:42

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

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:222

Prevents the modification of existing property attributes and values, and prevents the addition of new properties.

##### Type Parameters

###### T

`T` *extends* `Function`

##### Parameters

###### f

`T`

Object on which to lock the attributes.

##### Returns

`T`

#### Call Signature

> `static` **freeze**\<`T`, `U`\>(`o`): `Readonly`\<`T`\>

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:228

Prevents the modification of existing property attributes and values, and prevents the addition of new properties.

##### Type Parameters

###### T

`T` *extends* `object`

###### U

`U` *extends* `string` \| `number` \| `bigint` \| `boolean` \| `symbol`

##### Parameters

###### o

`T`

Object on which to lock the attributes.

##### Returns

`Readonly`\<`T`\>

#### Call Signature

> `static` **freeze**\<`T`\>(`o`): `Readonly`\<`T`\>

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:234

Prevents the modification of existing property attributes and values, and prevents the addition of new properties.

##### Type Parameters

###### T

`T`

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

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2019.object.d.ts:26

Returns an object created by key-value entries for properties and methods

##### Type Parameters

###### T

`T` = `any`

##### Parameters

###### entries

`Iterable`\<readonly \[`PropertyKey`, `T`\]\>

An iterable object that contains key-value entries for properties and methods.

##### Returns

`object`

#### Call Signature

> `static` **fromEntries**(`entries`): `any`

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2019.object.d.ts:32

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

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:310

Retrieves a registered function by its name or alias.

#### Parameters

##### name

`string`

The name or alias of the function to retrieve.

#### Returns

`ToolFunc`

The `ToolFunc` instance if found, otherwise `undefined`.

***

### getAllByTag()

> `static` **getAllByTag**(`tagName`): `ToolFunc`[]

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:327

Retrieves all registered functions that have a specific tag.

#### Parameters

##### tagName

`string`

The tag to search for.

#### Returns

`ToolFunc`[]

An array of matching `ToolFunc` instances.

***

### getByModel()

> `static` **getByModel**(`modelName?`): `undefined` \| `LLMProvider`

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:339

#### Parameters

##### modelName?

`string`

#### Returns

`undefined` \| `LLMProvider`

***

### getByTag()

> `static` **getByTag**(`tagName`): `undefined` \| `ToolFunc`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:321

Finds the first registered function that has a specific tag.

#### Parameters

##### tagName

`string`

The tag to search for.

#### Returns

`undefined` \| `ToolFunc`

The first matching `ToolFunc` instance, or `undefined` if none is found.

***

### getCurrentProvider()

> `static` **getCurrentProvider**(): `undefined` \| `LLMProvider`

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:340

#### Returns

`undefined` \| `LLMProvider`

***

### getFunc()

> `static` **getFunc**(`name`): `any`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:356

Retrieves a bound, runnable function reference for a registered function.
This reference is suitable for execution with an object of named parameters.

#### Parameters

##### name

`string`

The name of the function.

#### Returns

`any`

A bound function reference, or `undefined` if not found.

***

### getFuncWithPos()

> `static` **getFuncWithPos**(`name`): `any`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:379

Retrieves a bound, runnable function reference for a registered function.
This reference is suitable for execution with positional arguments.

#### Parameters

##### name

`string`

The name of the function.

#### Returns

`any`

A bound function reference, or `undefined` if not found.

***

### getOwnPropertyDescriptor()

> `static` **getOwnPropertyDescriptor**(`o`, `p`): `undefined` \| `PropertyDescriptor`

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:175

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

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2017.object.d.ts:48

Returns an object containing all own property descriptors of an object

#### Type Parameters

##### T

`T`

#### Parameters

##### o

`T`

Object that contains the properties and methods. This can be an object that you created or an existing Document Object Model (DOM) object.

#### Returns

\{ \[P in string \| number \| symbol\]: TypedPropertyDescriptor\<T\[P\]\> \} & `object`

***

### getOwnPropertyNames()

> `static` **getOwnPropertyNames**(`o`): `string`[]

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:182

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

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2015.core.d.ts:319

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

Defined in: property-manager.js/lib/advance.d.ts:10

get all properties descriptor include inherited.

#### Returns

`PropDescriptors`

***

### getPrototypeOf()

> `static` **getPrototypeOf**(`o`): `any`

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:167

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

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:333

Checks if any registered function has a specific asynchronous feature.

#### Parameters

##### feature

`AsyncFeatureBits`

The async feature bit to check for.

#### Returns

`boolean`

`true` if the feature is present in any function, otherwise `false`.

#### Inherited from

`CancelableAbility.hasAsyncFeature`

***

### is()

> `static` **is**(`value1`, `value2`): `boolean`

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2015.core.d.ts:332

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

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:258

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

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:252

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

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:246

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

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:264

Returns the names of the enumerable string properties and methods of an object.

##### Parameters

###### o

`object`

Object that contains the properties and methods. This can be an object that you created or an existing Document Object Model (DOM) object.

##### Returns

`string`[]

#### Call Signature

> `static` **keys**(`o`): `string`[]

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2015.core.d.ts:325

Returns the names of the enumerable string properties and methods of an object.

##### Parameters

###### o

Object that contains the properties and methods. This can be an object that you created or an existing Document Object Model (DOM) object.

##### Returns

`string`[]

***

### list()

> `static` **list**(): `Funcs`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:315

Returns the complete map of all registered functions.

#### Returns

`Funcs`

The map of `ToolFunc` instances.

***

### preventExtensions()

> `static` **preventExtensions**\<`T`\>(`o`): `T`

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:240

Prevents the addition of new properties to an object.

#### Type Parameters

##### T

`T`

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

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:398

Registers a new tool function.

##### Parameters

###### name

`string`

The name of the function.

###### options

`FuncItem`

The function's configuration.

##### Returns

`boolean` \| `ToolFunc`

The new `ToolFunc` instance, or `false` if a function with that name already exists.

#### Call Signature

> `static` **register**(`func`, `options`): `boolean` \| `ToolFunc`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:399

Registers a new tool function.

##### Parameters

###### func

`Function`

The function implementation.

###### options

`FuncItem`

The function's configuration.

##### Returns

`boolean` \| `ToolFunc`

The new `ToolFunc` instance, or `false` if a function with that name already exists.

#### Call Signature

> `static` **register**(`name`, `options?`): `boolean` \| `ToolFunc`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:400

Registers a new tool function.

##### Parameters

###### name

The name of the function.

`string` | `Function` | `ToolFunc` | `FuncItem`

###### options?

`FuncItem`

The function's configuration.

##### Returns

`boolean` \| `ToolFunc`

The new `ToolFunc` instance, or `false` if a function with that name already exists.

***

### run()

> `static` **run**(`name`, `params?`): `Promise`\<`any`\>

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:341

Asynchronously executes a registered function by name with named parameters.

#### Parameters

##### name

`string`

The name of the function to run.

##### params?

`any`

The parameters object for the function.

#### Returns

`Promise`\<`any`\>

A promise that resolves with the function's result.

#### Throws

If the function with the given name is not found.

***

### runSync()

> `static` **runSync**(`name`, `params?`): `any`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:349

Synchronously executes a registered function by name with named parameters.

#### Parameters

##### name

`string`

The name of the function to run.

##### params?

`any`

The parameters object for the function.

#### Returns

`any`

The result of the function's execution.

#### Throws

If the function with the given name is not found.

***

### runWithPos()

> `static` **runWithPos**(`name`, ...`params`): `Promise`\<`any`\>

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:364

Asynchronously executes a function using positional arguments.

#### Parameters

##### name

`string`

The name of the function to run.

##### params

...`any`[]

Positional arguments to pass to the function.

#### Returns

`Promise`\<`any`\>

A promise that resolves with the function's result.

#### Throws

If the function with the given name is not found.

***

### runWithPosSync()

> `static` **runWithPosSync**(`name`, ...`params`): `any`

Defined in: @isdk/ai-tools/packages/ai-tool/dist/index-BLW3R7VS.d.ts:372

Synchronously executes a function using positional arguments.

#### Parameters

##### name

`string`

The name of the function to run.

##### params

...`any`[]

Positional arguments to pass to the function.

#### Returns

`any`

The result of the function's execution.

#### Throws

If the function with the given name is not found.

***

### seal()

> `static` **seal**\<`T`\>(`o`): `T`

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es5.d.ts:216

Prevents the modification of attributes of existing properties, and prevents the addition of new properties.

#### Type Parameters

##### T

`T`

#### Parameters

##### o

`T`

Object on which to lock the attributes.

#### Returns

`T`

***

### setCurrentProvider()

> `static` **setCurrentProvider**(`name`): `void`

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:341

#### Parameters

##### name

`string`

#### Returns

`void`

***

### setPrototypeOf()

> `static` **setPrototypeOf**(`o`, `proto`): `any`

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2015.core.d.ts:339

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

Defined in: @isdk/ai-tools/packages/ai-tool-llm/dist/index.d.ts:342

Unregisters a function by its name, also removing any associated aliases.

#### Parameters

##### name

`string`

The name of the function to unregister.

#### Returns

`undefined` \| `ToolFunc`

The unregistered `ToolFunc` instance, or `undefined` if it was not found.

***

### values()

#### Call Signature

> `static` **values**\<`T`\>(`o`): `T`[]

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2017.object.d.ts:24

Returns an array of values of the enumerable own properties of an object

##### Type Parameters

###### T

`T`

##### Parameters

###### o

Object that contains the properties and methods. This can be an object that you created or an existing Document Object Model (DOM) object.

\{\[`s`: `string`\]: `T`; \} | `ArrayLike`\<`T`\>

##### Returns

`T`[]

#### Call Signature

> `static` **values**(`o`): `any`[]

Defined in: @isdk/ai-tools/node\_modules/.pnpm/typescript@5.7.3/node\_modules/typescript/lib/lib.es2017.object.d.ts:30

Returns an array of values of the enumerable own properties of an object

##### Parameters

###### o

Object that contains the properties and methods. This can be an object that you created or an existing Document Object Model (DOM) object.

##### Returns

`any`[]
