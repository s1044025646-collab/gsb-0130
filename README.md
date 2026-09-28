# 中文词性标注与序列模型评测后端

纯本地实现的一阶隐马尔可夫模型（HMM）中文词性标注后端。输入为**已分词**的句子，
自行统计句首 / 句尾 / 转移 / 发射概率，在对数空间用 Viterbi 解码。无网页、不下载
预训练模型、不调用任何外部语言服务；仅包含 API、CLI、合成语料夹具与自动化测试。

## 依赖

- Node.js >= 22（内置 `node:sqlite` 与 `fetch`；开发环境为 Node 25）
- TypeScript、tsx、@types/node（devDependencies，安装命令见下）

```powershell
npm install
```

## 模型说明

- **句首** `π(y)`、**转移+句尾联合归一** `P(y_t|y_{t-1})` 与 `P(STOP|y_{t-1})`、
  **发射** `P(x|y)` 均使用加一（Laplace，平滑量可配置）平滑，每个分布严格归一。
- 句子之间不产生任何转移统计；空句合法并返回空路径、分数 0。
- Viterbi 全程在对数空间计算，500 词全未知词长句分数仍有限，无下溢。
- 同分路径按标签字典序确定性打平，结果可重复。
- 未知词策略：
  - `uniform`：未知词在标签集上形成总和为 1 的均匀分布（跨标签归一）；
  - `closed`：发射表为 UNK 保留加一概率，每个标签的发射分布自身归一。
  - 未知词只会得到较小概率，绝不会让整句静默失败。
- 词典与全部统计**只来自训练划分**，验证集词汇无法进入模型词汇（测试中有泄漏防护用例）。
- 不允许的标签返回 `TAG_NOT_IN_MODEL`；允许标签集为空返回 `NO_FEASIBLE_PATH`。
- 相同语料重复导入幂等（SHA-256 指纹去重，不重复计数）；语料修订生成新版本，
  模型参数按参数指纹复用，旧模型持久化保存并可重现历史预测。

## 合成夹具

`fixtures/corpus.ts`（构建脚本 `scripts/write-fixtures.ts`，产物
`fixtures/synthetic-zh.json`）覆盖：

- 同词异性（`锁/n`、`锁/v`，`计划/计划`）
- 未知词、单词句、空句、罕见标签（叹词 `e`）
- 23 词长句（压力 / 数值稳定）
- 训练验证重复样本（完全相同的句子出现两次，校验去重）

重新生成夹具：

```powershell
npx tsx scripts/write-fixtures.ts
```

## CLI

数据库默认在项目内 `data/app.sqlite`，可用环境变量 `HMM_DB_PATH` 覆盖。

```powershell
# 校验 / 幂等导入 / 列表
npx tsx src/cli/index.ts validate fixtures/synthetic-zh.json
npx tsx src/cli/index.ts import   fixtures/synthetic-zh.json
npx tsx src/cli/index.ts corpora

# 固定种子划分
npx tsx src/cli/index.ts split 1 --seed 42 --ratio 0.8

# 训练（smoothing >= 0；unknown = uniform | closed）
npx tsx src/cli/index.ts train 1 --smoothing 1 --unknown uniform
npx tsx src/cli/index.ts models

# 标注与逐步路径解释（可用 --allowed-tags 限制标签）
npx tsx src/cli/index.ts tag 1 --sentence "我 爱 北京"
npx tsx src/cli/index.ts explain 1 --sentence "我 爱 一个陌生词"

# 评测并导出逐标签 P/R/F1 与混淆矩阵
npx tsx src/cli/index.ts evaluate 1 --part validation --out data/report.json

# 模型比较
npx tsx src/cli/index.ts compare 1 2
```

关键命令会打印操作前后状态（如模型总数、训练 / 验证句数、是否复用已有版本）。

## HTTP API

仅绑定 `127.0.0.1`，端口由系统自动分配（0 = 空闲端口），启动日志会打印实际端口。

```powershell
npm start                 # 前台运行，Ctrl+C 停止
powershell scripts/serve.ps1   # 后台运行（只管理本项目进程，pid 写入 data/server.pid）
powershell scripts/stop.ps1    # 停止后台服务
```

路由（JSON；分页参数 `limit`(1-200) / `offset`）：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET  | `/health` | 健康检查 |
| POST | `/corpora` | 导入语料（幂等） |
| GET  | `/corpora?limit=&offset=` | 分页列出语料版本 |
| GET  | `/corpora/:id` | 语料与句子明细 |
| POST | `/corpora/:id/splits` | 固定种子划分 `{seed, trainRatio}` |
| POST | `/models` | 训练 `{splitId, smoothing, unknownStrategy}` |
| GET  | `/models` | 分页列出模型 |
| POST | `/models/:id/tag` | 标注 `{tokens, allowedTags?}`，返回标签、分数、逐步解释 |
| POST | `/models/:id/reports` | 评测 `{part: "validation"|"train"}` |
| GET  | `/reports?modelId=&limit=&offset=` | 分页查询报告 |
| GET  | `/reports/:id` | 报告详情（逐标签指标、混淆矩阵） |

错误统一为 `{ "error": { "code", "message", "details?" } }`，HTTP 状态：
400 校验类（`INVALID_JSON` / `INVALID_CORPUS` / `INVALID_SENTENCE` /
`INVALID_BODY` / `INVALID_QUERY` / `INVALID_PARAMS` / `CLI_USAGE`），
404 `NOT_FOUND`，422 `TAG_NOT_IN_MODEL` / `NO_FEASIBLE_PATH`。

## 测试与演示

```powershell
npm test     # node:test，23 个用例
npm run demo # 可重复的端到端演示（自动重建 data/demo，含 CLI + HTTP）
npm run build
```

测试覆盖：概率分布归一、对数空间数值稳定（500 未知词）、短句穷举核对 Viterbi、
确定性同分规则、未知词与空句、固定种子划分、训练集泄漏防护、重复导入幂等、
语料修订版本隔离与旧模型重现、评测逐标签指标与混淆矩阵、事务回滚、重启持久化、
API 参数校验 / 错误码 / 分页、CLI 全流程。时间通过可注入的 `Clock` 控制。

## 数据与清理

所有数据库、日志与临时文件都在项目内 `data/` 目录；删除该目录即可彻底清理：

```powershell
powershell scripts/stop.ps1
Remove-Item -Recurse -Force data
```
