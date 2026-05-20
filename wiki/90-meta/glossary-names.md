# 论文和模型名称词源

LLM 论文喜欢用缩写、神话人物、音乐术语、地名当名字，真实含义往往藏在附录里或根本不解释。

| 名称 | 全称 / 词源 | 梗或来源 |
|---|---|---|
| **Alpaca** | Stanford Alpaca | 羊驼，骆驼科动物，LLaMA 的亲戚——Stanford 用 Self-Instruct 方法在 LLaMA 上做指令微调，命名沿用"骆驼科"主题 |
| **BPE** | **B**yte **P**air **E**ncoding | 数据压缩算法的名称，原本由 Philip Gage 于 1994 年提出用于压缩，被 Sennrich et al. 2016 引入 NLP 做 subword tokenization |
| **ccNet** | **C**ommon **C**rawl **Net**work（pipeline）| 直接描述：用于处理 Common Crawl 数据的网络（管线），无特别梗，功能命名 |
| **Chinchilla** | 无缩写，就是"龙猫/毛丝鼠" | DeepMind 用来修正 Gopher 的模型。Chinchilla（毛丝鼠）比 Gopher（囊地鼠）小——对应论文的核心发现：更小的模型用更多数据训练，比 Gopher 这样"过大欠训"的模型更高效 |
| **DETR** | **DE**tection **TR**ansformer | 功能描述性缩写，Facebook AI 2020。不是词汇，纯拼写：DE=Detection，TR=Transformer，去掉中间的 tec 和 ans，保留头尾辅音让发音顺畅（读 "dee-ter"）|
| **DCLM** | **D**ata**C**omp for **L**anguage **M**odels | DataComp 是一个数据竞赛框架（"DataComp-for-LM"），LM 前面的 DC 对应 "DataComp"。DataComp 本身是"Data Competition"的缩写，强调用竞赛方式找最优数据过滤策略 |
| **Deita** | **D**ata-**E**fficient **I**nstruction **T**uning for **A**lignment | "data"的变形拼写，暗示数据效率是核心。deita 不是已有词汇，纯粹是拼写游戏 |
| **Dolma** | **D**ata f**o**r **O**pen **L**anguage **M**odels' **A**ppetite | AI2 的 3T token 开放语料库。dolma 也是中东/地中海料理中的"填馅蔬菜"（如用葡萄叶包的米饭），契合"喂饱语言模型"的隐喻 |
| **DoReMi** | **Do**main **Re**weighting with **Mi**nimax optimization | 音乐 solfège（唱名法）的前三个音节 Do-Re-Mi，对应算法的三个核心步骤：训练参考模型（Do）、用 minimax 优化权重（Re）、用优化后权重训练大模型（Mi） |
| **DPO** | **D**irect **P**reference **O**ptimization | RLHF 的简化替代，"direct"意思是不需要单独训练 reward model，直接在 LM 上优化偏好目标 |
| **FineWeb** | Fine（好的）+ Web（网络爬取）| HuggingFace 的高质量网页数据集，"fine"双关：精细过滤 + 高质量，无深层梗 |
| **Gopher** | 无缩写，就是"囊地鼠" | DeepMind 的 280B 模型。DeepMind 用动物命名自家模型系列（Flamingo、Chinchilla、Gopher）。囊地鼠（gopher）是北美的一种挖地穴的小动物，无明显隐喻，纯粹是动物主题 |
| **LIMA** | **L**ess **I**s **M**ore for **A**lignment | 来自秘鲁首都利马（Lima），同时对应论文主张"少量高质量数据就够"。meta 学术梗：LIMA 是对 RLHF 繁复流程的反叛，"Less is More"这个短语本身来自建筑师 Mies van der Rohe |
| **LIMO** | **L**ess **I**s **M**ore for **R**easoning（作者用 O 结尾构成呼应）| 对 LIMA 的致敬/呼应——LIMA 是对齐领域的"少即是多"，LIMO 是推理领域的"少即是多"。limo 也是豪华轿车（limousine 缩写），但论文没有显式用这个梗 |
| **LLaMA** | **L**arge **L**anguage **M**odel **M**eta **A**I | Meta 的开源基础模型系列，注意大写的 LLaMA，不是骆驼（llama），但发音和骆驼（llama）相同，确有骆驼隐喻——骆驼以耐力著称，对应"高效/开源"的定位 |
| **MagPie** | **Mag**netic **P**rompt-**I**nduced **E**lution（作者解释）| 喜鹊（magpie），善于模仿人类语言——恰好契合"用 aligned LLM 模仿用户提问"的方法。喜鹊也以收集闪亮东西闻名，对应"从模型里提取有价值数据" |
| **MCTS** | **M**onte **C**arlo **T**ree **S**earch | 蒙特卡洛树搜索。蒙特卡洛（Monaco 的卡西诺城市）是"靠随机模拟估计期望值"方法的统称，来源于 20 世纪中期核武器模拟计算中的赌博隐喻 |
| **MFU** | **M**odel **F**LOPs **U**tilization | 训练效率度量，PaLM 论文（2022）提出。FLOPs = floating point operations，MFU 是对峰值理论算力的利用率 |
| **OLMo** | **O**pen **L**anguage **M**odel | AI2 的完全开放语言模型（模型+数据+代码全开放），olmo 也是西班牙语"榆树"（elm），AI2 的其他项目也有植物命名偏好 |
| **PaLM** | **P**athways **L**anguage **M**odel | Google 的 540B 模型，用 Pathways 分布式系统训练，"palm"也是棕榈树，Google 产品常见植物命名风格（TPU Pod 的代号也用植物） |
| **Phi** | 希腊字母 φ | Microsoft Research 的小模型系列（phi-1, phi-2, phi-3）。φ 在数学里常代表"黄金比例"或各类函数，隐喻"小而精"的比例感。phi-1 专注代码，phi-2/phi-3 扩展到通用推理 |
| **PPO** | **P**roximal **P**olicy **O**ptimization | RL 中的策略优化算法，"proximal"强调每步更新不能离上一步太远（近端约束），OpenAI 2017 年提出 |
| **Quiet-STaR** | **St**eps **a**nd **R**ationale（原 STaR 论文），Quiet 代表"静默思考" | STaR（Self-Taught Reasoner，2022）的扩展版，Quiet 前缀表示推理在 `<think>` token 内部进行，对最终输出"静默"不可见。STaR 本身命名来自"star"（明星），隐喻"自我成长" |
| **ReStar** | **Re**inforced **S**elf-**T**r**a**ining with **R**easoning | 对 STaR 的强化学习扩展，Re- 前缀表示"用 RL 重新训练"，也暗示"再次成为明星（ReSTAR）"的双关 |
| **RLHF** | **R**einforcement **L**earning from **H**uman **F**eedback | 描述性缩写，无梗。由 Christiano et al. 2017 提出，InstructGPT/ChatGPT 的对齐核心技术 |
| **Self-Instruct** | 无缩写，描述性命名 | 字面意思：模型用自己生成的指令来对齐自己（self instruction）。这是该领域命名最透明的论文之一，没有刻意造梗 |
| **MTR** | **M**otion **TR**ansformer | 描述性缩写，motion prediction + transformer。论文全名"Motion Transformer with Global Intention Localization and Local Movement Refinement"，MTR 是模型简称 |
| **UniAD** | **Uni**fied **A**utonomous **D**riving | 描述性缩写，"统一的自动驾驶"——把感知、预测、规划统一进一个网络。论文原标题是"Planning-oriented Autonomous Driving"，UniAD 是模型的名称 |
| **Vicuna** | Vicuña（骆马）| 南美骆驼科，LLaMA 的另一亲戚——LMSYS 用 ShareGPT 对话数据微调 LLaMA，骆驼族谱继续延伸 |
| **PLUTO** | **P**rediction and p**L**anning **U**sing **T**ransf**O**rmers | 拼写游戏：取 Prediction、pLanning、Using、Transformers 的首字母，拼出冥王星（Pluto）——太阳系边缘的矮行星，隐喻"把学习规划推向极限"（Pushing the Limit）；也和论文副标题"Pushing the Limit..."中的 PL 呼应 |
| **Occ3D** | **3D Occ**upancy Prediction | 描述性命名，直接说明任务：3D 空间的每个 voxel 占据状态（occupancy）预测。Occ 是业内对该任务的简称 |
| **TPVFormer** | **T**ri-**P**erspective **V**iew **Former** | TPV = Tri-Perspective View（三视图），Former = Transformer 后缀（行业命名惯例）。三视图指顶视图(HW)、侧视图(DH)、前视图(WD)三个互相垂直的平面 |
| **VoxFormer** | **Vox**el Trans**former** | Vox 来自 voxel（体素），Former 是 Transformer 后缀。强调在 voxel 表示上做 Transformer 操作 |
| **MonoScene** | **Mono**cular **Scene** completion | 描述性命名：单目（monocular，即单张 RGB 图像）做 3D 场景完成（scene completion）。Mono 直接点明和多模态/LiDAR 方法的区别 |
| **CTF-Occ** | **C**oarse-**T**o-**F**ine **Occ**upancy | Occ3D 论文提出的附带模型，描述其渐进式精化策略（先粗后细，coarse-to-fine pyramid 结构） |
| **BEV** | **B**ird's-**E**ye **V**iew | 鸟瞰图，自动驾驶感知领域常用的 2D 投影表示，把 3D 世界投影到地面平面的俯视图 |
| **MapTR** | **Map** **TR**ansformer | 描述性缩写，Map = HD Map 构建任务，TR = Transformer。同时暗含"Map Tracker"含义——在 BEV 空间追踪地图元素的轮廓 |
| **HDMapNet** | **HD Map** **Net**work | 描述性命名，专门针对 HD Map（高精地图）构建的神经网络，强调输出是可下游使用的高精度地图，区别于普通语义分割 |
| **VectorMapNet** | **Vector** **Map** **Net**work | 强调"向量化"（vectorized）输出——把地图元素表示为连续折线/多边形而非栅格像素，是 MapTR 之前的端到端向量化建图先驱 |
| **GKT** | **G**eometry-guided **K**ernel **T**ransformer | 用几何投影关系引导 BEV 特征聚合的 Transformer，MapTR 的默认 2D→BEV 变换模块 |
| **VAD** | **V**ectorized scene representation for **A**utonomous **D**riving | 描述性缩写，强调把驾驶场景（agent + 地图元素）全部向量化后进行端到端规划 |
