# zh-pos-hmm — 中文词性标注与序列模型评测后端

从零实现的、**完全本地运行**的中文词性标注（POS tagging）后端。使用
TypeScript + Node.js 内置 SQLite（`node:sqlite`）实现，自训练**一阶隐马尔可夫
模型（HMM）**并用 **Viterbi** 在对数空间解码。

- 不提供网页、不下载预训练模型、不调用任何外部语言服务。
- 仅提供：HTTP API、命令行（CLI）、合成标注语料夹具、自动化测试。
- 数据与临时文件全部位于项目内 `.data/`；HTTP 仅绑定 `127.0.0.1` 且使用自动端口。

## 依赖与环境

- Node.js `>= 22.5`（开发与验证使用 Node 25；依赖内置 `node:sqlite`）。
- 安装依赖：

```powershell
npm install
```

仅有的运行期依赖是 Node 内置模块；开发依赖为 `typescript`、`tsx`、`@types/node`。

## 常用命令

```powershell
npm run typecheck   # 仅类型检查
npm test            # 运行全部自动化测试（node:test）
npm run demo        # 可重复的端到端演示（独立 .data/demo.sqlite）

# CLI（可用 npm run cli -- <子命令> 或 npx tsx src/main.ts <子命令>）
npx tsx src/main.ts help
```

## CLI 工作流

```powershell
npx tsx src/main.ts init-db
npx tsx src/main.ts corpus-add --name 示例语料
npx tsx src/main.ts import  --corpus 1 --file fixtures/corpus.json
npx tsx src/main.ts split   --version 1 --seed 42 --ratio 0.25
npx tsx src/main.ts train   --version 1 --seed 42 --ratio 0.25 --alpha 1 --unknown laplace
npx tsx src/main.ts tag     --model 1 --file fixtures/tag-input.json
npx tsx src/main.ts explain --model 1 --file fixtures/tag-input.json
npx tsx src/main.ts evaluate --model 1
npx tsx src/main.ts compare  --models 1,2
```

导入按语料**内容指纹**去重：重复导入复用同一版本，绝不重复增加训练计数；
语料内容修订会生成新版本，新版本训练出独立模型，旧模型行仍可被加载并
**重现历史预测**（模型持久化的是训练计数与参数，预测时按同一公式重建）。

关键操作（建库、导入、划分、训练）都会在终端打印操作前后的数量状态。

## 启动与停止（仅本项目进程）

```powershell
npm start
# 等价于：node --experimental-strip-types src/main.ts serve
# 或：npx tsx src/main.ts serve
```

启动后监听 `http://127.0.0.1:<自动端口>`，端口与 PID 写入
`.data/server.port` 与 `.data/server.pid`。停止：

```powershell
npx tsx src/main.ts stop
```

`stop` 只读取本项目 `.data/server.pid` 并终止其中记录的进程，随后清理本项目的
pid/port 文件，不触碰其它进程。

## HTTP API

所有请求/响应均为 JSON，错误统一为
`{ "error": { "code", "message", "details" } }`，带明确错误码与 HTTP 状态：

| 错误码 | 状态 | 含义 |
| --- | --- | --- |
| `VALIDATION` | 400 | 参数非法（含 JSON 解析失败） |
| `BAD_TAG` | 400 | 指定了模型不认识的标签 / 空标签白名单 |
| `NO_PATH` | 400 | 当前标签与平滑下无概率为正的路径 |
| `NOT_FOUND` | 404 | 资源不存在 / 未知路由 |
| `CONFLICT` | 409 | 划分与语料版本不匹配 |
| `EMPTY_CORPUS` | 422 | 导入数组为空 |
| `INSUFFICIENT_DATA` | 422 | 训练/验证集为空或无标签 |

端点：

- `GET  /health`
- `POST /api/corpora`、`GET /api/corpora`、`GET /api/corpora/:id`
- `POST /api/corpora/:id/import`（body: `{ sentences: [{tokens,tags}] }`，幂等）
- `GET  /api/corpora/:id/versions`
- `GET  /api/versions/:id/sentences?limit=&offset=`（分页）
- `POST /api/versions/:id/split`（`{seed, valRatio}`，固定划分，幂等）
- `POST /api/models/train`（`{versionId, seed, valRatio, alpha, unknownStrategy}`）
- `GET  /api/models?corpusId=&limit=&offset=`（分页）、`GET /api/models/:id`
- `POST /api/tag`（`{modelId, tokens, allowedTags?}`）
- `POST /api/models/:id/evaluate`（逐标签 P/R/F1 + 混淆矩阵，落库一条报告）
- `GET  /api/models/:id/reports?limit=&offset=`（分页）
- `POST /api/models/compare`（`{modelIds:[...]}`，至少两个）

示例（PowerShell）：

```powershell
$port = Get-Content .data\server.port
Invoke-RestMethod "http://127.0.0.1:$port/health"
Invoke-RestMethod "http://127.0.0.1:$port/api/tag" -Method Post -ContentType 'application/json' `
  -Body '{"modelId":1,"tokens":["企鹅","吃","磷虾"]}'
```

## 模型与算法约定

- **计数来源严格限定训练集**：句首、句尾、相邻标签转移、标签→词发射，以及
  词表本身都只统计训练句；句子之间**不会**误计跨句转移；空句不贡献任何计数。
- **平滑与归一**（`alpha >= 0`）：
  - 句首 / 句尾 / 转移：在 `|T|` 个标签上加 `alpha`，每个条件分布和为 1。
  - 发射 `laplace`：在 `|V|+1` 个符号（含预留 `UNK`）上加 `alpha`，每个标签
    的分布和为 1；未知词使用预留的 `UNK` 概率，因此**不会让整句静默失败**。
  - 发射 `closed`：已知词 MLE；未知词在标签集合上给均匀对数概率（不污染词表）。
- **Viterbi**：全程对数空间，避免长句下溢；返回最佳标签序列、完整路径分数
  （含句尾项）与逐位置解释（转移/发射/累计）。
- **确定性同分决胜**：标签按训练时字典序编号，更新只接受严格更优值，
  平分时保留编号更小的标签，重复运行结果一致。
- **不可行情形**：白名单含未知标签 → `BAD_TAG`；`alpha=0` 下无正概率路径
  （如某标签从不出现在句首）→ `NO_PATH`，并在 `details` 中给出上下文。

## 泄漏与版本隔离

- 固定种子划分把 `(seed, 句子指纹)` 哈希到 `[0,1)`，内容相同的重复句哈希
  相同，因此**绝不会一条在训练、另一条在验证**；划分结果按
  `(版本, seed, ratio)` 持久化复用。
- 词表与发射计数只来自训练集，验证集专有词在模型中恒为未知词（测试覆盖）。
- 语料修订产生新版本与新模型；旧模型持久化行不被覆盖，可随时重现旧预测。

## 合成语料夹具

`fixtures/corpus.ts`（及导出的 `fixtures/corpus.json`）覆盖：
同词异性（“计划/花/打”）、未知词（`validationOnly`）、单词句、空句、
罕见标签 `X`、长句，以及逐字重复样本。标签集：
`N V A ADV P NUM M PRON CONJ PUNCT X`。

## 测试覆盖

`npm test` 运行 21 个测试，覆盖：

- 各分布概率归一、对数空间数值稳定（200 词长句）。
- 短句**全枚举**与 Viterbi 最优路径/分数逐项核对。
- 未知词不致命、确定性同分决胜、`BAD_TAG`/`NO_PATH` 失败路径。
- 重复导入幂等、重复训练幂等、事务失败回滚不留半成品。
- 固定种子划分的跨集重复防护、训练/验证词汇泄漏防护、版本隔离与旧模型重现。
- 关闭重开后的持久化、可重复迁移，以及 HTTP API 端到端、参数校验与错误码。
