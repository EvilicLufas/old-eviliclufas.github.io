---
title: RASD Final
date: 2019-12-18 17:44:19
tags: RASD 3rd_year 
---

# RASD Final Review

##### 1. 迭代增量开发

1.Define iterative incremental development. Give five examples of iterative incremental processes used in software development.

> 1.定义迭代增量开发。给出五个示例，说明软件开发中使用的迭代增量过程。

迭代增量量模型是软件开发过程中、常用的开发模型。
迭代是在实现软件的每⼀一功能时反复求精的过程，是提升软件质量量的过程，是从模糊到清晰的过程。在连续的
迭代中增加细节，必要时引入变更和改进。（是功能越来越好）
增量是强调软件在发布不不同的版本时，每次都多发布⼀一点点，是软件功能数量量渐增地发布的过程。增量版本保
持了用户的满意度，为尚在开发中的模块提供重要的反馈。（使产品越来越好）
例子：
1.螺旋模型
2.Rational 统⼀一过程（RUP）
3.模型驱动的体系结构（MDA）
4.敏敏捷开发过程
5.面向方面的软件开发(AOP，Aspect-Oriented programming)

> Iterative incremental model is a commonly used development model in software development.
> Iteration is the process of refining repeatedly while implementing each function of the software, the process of improving the quality of the software, and the process from obscurity to clarity.
> Add details in successive iterations and introduce changes and improvements if necessary.
> (It's getting better and better.) Incremental means emphasizing that when software releases different versions, it releases a little more each time. It is a process of increasing the number of software features.
> Incremental versions maintain user satisfaction and provide important feedback for modules that are still under development.
> (Make the product better and better) 
>
> Example: 
>
> 1. Spiral model 2. Rational Unified Process (RUP) 3. Model-driven architecture (MDA) 4. Smart development process 5. AOP，aspect oriented programming]



##### 2. COBIT

2.What is COBIT? How does it differ from ISO9000 and ITIL? What are the four domains COBIT groups IT-related efforts into?

> 2.什么是COBIT？它与ISO9000和ITIL有何区别？COBIT组与IT相关的工作涉及四个领域？



COBIT(Control Objectives for Information and related Technology) 是目前国际上通用的信息系统审计的标
准，是⼀一个服从框架。
ISO 9000和ITIL都是过程标准，ISO 9000应用于质量量管理理和过程以生产优质产品，ITIL致力于解决方案交付和
管理理的操作方面；COBIT则是一个产品标准，它致力力于解决方案管理理的控制方面，侧重于一个组织需要做什什么
而非如何去做。
COBIT将相关的IT工作归为4个领域：
规划与组织（Plan and Organize）、获取与实现（Acquire and Implement）、交付与支持（Deliver and
Support）、监控（Monitor and Evaluate）

>  COBIT (Control Objectives for Information and Related Technology) is currently the international standard for information system auditing and is a compliance framework.
> ISO 9000 and ITIL are process standards. ISO 9000 applications are used for quality management and processes to produce high-quality products. ITIL is committed to the operational aspects of solution delivery and management. COBIT is a product standard.
> Regarding the control of solution management, the focus is on what an organization needs to do rather than how to do it.
> COBIT classifies related IT work into 4 areas: Plan and Organize, Acquire and Implement, Deliver and Support, Monitor and Evaluate

##### 3. UML

3.What is UML? What must a process do in order to adopt UML? Where is UML deficient?
Name the three categories of models used in UML and describe what they do.

> 3.什么是UML？为了采用UML，流程必须做什么？UML缺乏之处在哪里？
> 命名UML中使用的三类模型，并描述它们的作用。



UML（Unified Modeling Language 统一建模语言）是用来对软件密集系统进行行可视化建模的一种语言。
采用 UML 的过程必须支持一种面向对象的方法来进行软件生产（面向过程的对应的使用数据流图）。
缺点：不能表达非功能性需求，用例图是描述用户功能需求的工具，对于可靠性、性能等非功能性需求无能为
力。
UML 的模型可分为3组：
状态模型：描述静态数据结构
行为模型：描述对象协作
状态改变模型：描述随着时间的推移，系统所允许的状态

>  UML (Unified Modeling Language) is a language for visually modeling software-intensive systems.
>  UML processes must support an object-oriented approach to software production (process-oriented corresponding use of data flow diagrams).
>  Disadvantages: I can't express non-functional requirements. I use a map to describe the user's functional requirements. I can't do anything about non-functional requirements such as reliability and performance.
>  UML models can be divided into three groups: 
>
> Structural modeling: - It captures the static features of a system.
>
> Behavioral modeling: - It describes the interaction within the system. It represents the interaction among the structural diagrams. Structural model represents the framework for the system and this framework is the place where all other components exist.
>
> Architectural modeling: - It represents the overall framework of the system. It contains both structural and behavioral elements of the system. Architectural model can be defined as the blueprint of the entire system. 

3.What is UML? What must a process do in order to adopt UML? Where is UML deficient? Name the three categories of models used in UML and describe what they do.
UML（Unified Modeling Language 统⼀建模语⾔）是⽤来对软件密集系统进⾏可视化建模的⼀种语⾔。 采⽤ UML 的过程必须⽀持⼀种⾯向对象的⽅法来进⾏软件⽣产（⾯向过程的对应的使⽤数据流图）。 缺点：不能表达⾮功能性需求，⽤例图是描述⽤户功能需求的⼯具，对于可靠性、性能等⾮功能性需求⽆能为 ⼒。
UML 的模型可分为3组： 状态模型：描述静态数据结构 ⾏为模型：描述对象协作 状态改变模型：描述随着时间的推移，系统所允许的状态





4.Which CMM level of maturity is needed for the organization to be able to respond successfully to a crisis situation? Explain.
成熟度等级2：可重复级(Repeatable)。在这⼀级，有些基本的软件项⽬的管理⾏为、设计和管理技术是基于 相似产品中的经验，故称为“可重复”。在这⼀级采取了⼀定措施，这些措施是实现⼀个完备过程所必不可缺少 的第⼀步。典型的措施包括仔细地跟踪费⽤和进度。不像在第⼀级那样，在危机状态下⽅⾏动，管理⼈员在问 题出现时便可发现，并⽴即采取修正⾏动，以防它们变成危机。关键的⼀点是，如没有这些措施，要在问题变 得⽆法收拾前发现它们是不可能的。在⼀个项⽬中采取的措施也可⽤来为未来的项⽬拟定实现的期限和费⽤计 划。
5.Based on your experiences with software products, how would you interpret the observation that the essence of software engineering is determined by inherent software

complexity, conformity, changeability, and invisibility? How would you explain these four factors? How is software engineering different from traditional engineering, such as civil or mechanical engineering?
软件的开发不仅仅存在⼀种因素影响着软件的开发过程，软件开发过程是⼀个⼗分困难的过程，存在四个基本 性质。 复杂性：软件规模的函数以及组成软件产品的部件之间相互依赖的函数。 ⼀致性：应⽤软件必须与其依赖的软硬件平台⼀致，并且必须与其现有信息系统保持⼀致以便于集成。 可变性：软件产品所描述的业务过程和需求经常发⽣变化，所以软件产品必须做出相应的改变以适应这些变化。
不可⻅性：⽣成输出的程序语句、⼆进制代码以及周边系统软件通常不可⻅。 ⾄于软件⼯程与传统⼯程的区别，我认为最⼤的不同在于相较于实体化、具象化的传统⾏业，软件的开发更加 抽象，难以对其总结开发经验，总是抱着解决问题的⼼态去完成的。
6.Recall the definition of a stakeholder. Is a software vendor or a technical support person a stakeholder? Explain.
利益相关者的定义指对系统产⽣影响或者被系统所影响的⼈，是在软件项⽬中存在利害关系的⼈ 所以软件供应商应该会被系统本身质量的优劣影响，质量差卖的少，供应商会受到影响，收益降低，所以是利
益相关者 技术⽀持⼈员回影响到系统本身，所以是利益相关者 软件供应商或技术⽀持⼈员显然符合这⼀定义，即主要的利益相关者是⽤户、系统所有者、分析员、设计员、 程序员等。
7.What does the acronym SWOT mean? Give a detailed overview of SWOT and how it would be applied in an organization. Use a diagram to explain your answer.

  


SWOT 指优势、缺陷、机会、威胁（strength、weakness、opportunity、threat）。SWOT ⽅法以调整组织的优势、劣势、机会和威胁的⽅式来进⾏信息系统开发项⽬的识别、分类、排序和选择。这是 ⼀个从确定组织使命开始的、⾃顶向下的⽅法。将与研究对象密切相关的各种主要内部优势、劣势和外部的机 会和威胁等，通过调查列举出来，并依照矩阵形式排列，然后⽤系统分析的思想，把各种因素相互匹配起来加 以分析，从中得出⼀系列相应的结论，⽽结论通常带有⼀定的决策性。
8.What are the three management levels? Consider a banking application that monitors the usage patterns of a credit card by its holder in order to block the card automatically when the bank suspects a misuse (theft, fraud, etc.). Which management level is addressed by such an application? Give reasons.
三个管理层次是策略级、战术级、操作级。 该应⽤程序解决的问题所属类型为操作级问题。 操作级关注员⼯⽇常活动和⽣产⽀持。
9.Explain what a On-line Transaction Processing System is. What is a transaction in terms of OLTP systems? Why are transactions necessary in databases?
OLTP systems(联机事务处理系统)是指利⽤计算机⽹络，将分布于不同地理位置的业务处理计算机设备或⽹络 与业务管理中⼼⽹络连接，以便于在任何⼀个⽹络节点上都可以进⾏统⼀、实时的业务处理活动或客户服务。 事务的提出主要是为了解决并发情况下保持数据⼀致性的问题。 ⽐⽅说⼀个客户给另外⼀个客户打钱，⼀个客户的账户余额增加⽽另外⼀个减少，如果在这过程中出错，则要 同时回滚两个的数据，保持了数据的⼀致性。
10.What is Process Hierarchy Modeling (PHM)? Is a Process Hierarchy Diagram part of Business Process Modeling Notation (BPMN)? Explain what a process is. Using diagrams explain the difference between composite and atomic processes.
过程层次建模是对由活动定义的有层次结构的业务过程建模。 BPMN 不⽀持过程的结构建模。 业务过程可能是⼿⼯操作的活动或者⾃动化服务。⼀个过程⾄少有⼀个输⼊流和⼀个输出流。过程获得控制， 主要通过将输⼊流转变为输出流来完成相应的活动。 原⼦过程（任务）不包含任何⼦过程。复合过程通过⼦过程来描述它的⾏为。 以下是图：

  


11.Explain what Business Process Modeling Notation (BPMN) is. Are there any alternatives to BPMN? Using diagrams describe the four basic categories of modeling elements in BPMN.
BPMN 专⻔⽤于对由活动定义的业务过程建模，这些活动能够产⽣对企业或其外部利益相关者有价值的事物。
UML 活动图是 BPMN 的⼀个替代者。
BPMN 的四种基本元素是流对象、连接对象、泳池和⼈⼯制品。
12.What are the three phases of solution envisioning? Describe each in detail.
解决⽅案构想过程的三个阶段：
① 业务能⼒探索：确定业务能⼒，即IT解决⽅案提交具体成果的能⼒；描述能⼒案例，即解决⽅案思路，为每
个能⼒⽣成⼀个业务案例。

  


② 解决⽅案能⼒构想：⽬的是将能⼒⽤例发展为解决⽅案概念（将业务环境作为输⼊，产⽣的未来新⼯作⽅ 法的构想作为输出），确保利益相关者对其意⻅保持⼀致；解决⽅案概念将业务环境作为输⼊，产⽣的未来新 ⼯作⽅法的构想作为输出。解决⽅案概念集中于最终的解决⽅案体系结构，并在解决⽅案构想研讨中得到发展。
③ 软件能⼒设计：取决于系统实现技术；开发软件能⼒体系结构、细化具有项⽬规划和⻛险分析的业务⽤例； 是软件建模的⼀项活动，为构建解决⽅案开发⾼层模型并制定计划。建模计划包括功能(功能性需求)、质量属 性(⾮功能性需求)和能⼒体系结构，显示⾼层软件构件之间的相互作⽤。
13.What is Requirements Elicitation? What does it involve? What artifacts does it produce? What is the difference between a functional and a non-functional requirement? Give an example of a functional and a non-functional requirement.
需求引导就是业务分析员通过咨询发现系统的需求。 它涉及客户和问题领域专家，需要领域知识和⾜够的经验。 最后得出客户所需要的系统确定的需求定义。 功能性需求需要从客户处获得，是系统期望的服务。例如：系统的范围，必要的业务功能，所需的数据结构。 ⾮功能性需求本质上不是⾏为的，⽽是系统开发和实现过程中的约束。例如：可⽤性，可复⽤性，可靠性，性 能，效率，适应性，其它约束。
14.Give four methods used in Requirements Elicitation. Describe each in detail.
需求引导的传统⽅法包括⾯谈、调查表、观察和研究业务⽂档。
1.与客户和领域专家⾯谈：⾯谈是发现事实和聚集信息的基本技术。⼤多数的⾯谈过程都是与客户⼀起进⾏的。 与客户⾯谈⼤多⽤来导出⽤例需求。如果业务分析员没有⾜够的领域知识的话，可以邀请领域专家⾯谈。与领 域专家的⾯谈经常是⼀个知识转换的过程，即对业务分析员来说是⼀个学习过程。⾯谈有两种基本形式：结构 化的和⾮结构化的。结构化⾯谈需要提前准备，有⼀个明确的⽇程，并且许多问题都是预先确定的。⾮结构化 ⾯谈更像⾮正式的会议，没有预定问题或预计的⽬的。
2.调查表：调查表是向很多客户收集信息的有效⽅法。它⼀般⽤来作为⾯谈的补充形式，⽽不是要替代它。调 查表应该设计得使回答问题尽量容易，特别应该避免开放式问题，⼤多数问题都应该是封闭式的。
3.观察：当客户不能有效地表达信息，或者只有⼀个完整的业务过程中的⽚段知识时，观察可能是有效的发现 事实的技术。要使观察具有代表性，观察应该持续较⻓的⼀段时间，在不同的时间段上和不同的⼯作负荷下挑 选时间进⾏。
4.⽂档和软件系统的研究：⽂档和软件系统的研究是发现⽤例需求和领域知识需求的宝贵技术。⽤例需求通过 研究已有的企业⽂档和系统表格或报告来发现。要研究的组织⽂档包括：业务表格、⼯作过程、职位描述、业 务计划等，要研究的系统表格和报表包括：计算机屏幕和报表，领域知识需求通过研究领域刊物和参考⼿册获 得。
15.Why is Requirements Negotiation and Validation needed?
客户的需求也许是重叠的（overlap）或者⽭盾的（conflict）。有些需求可能是模棱两可的（ambiguous）或 者不现实的，其他⼀些需求可能还没有发现。
由于这些原因，在形成需求⽂档之前，需要对需求进⾏协商与确认。
16.Define requirement risk and priorities. Give five categories of risk and give an example of each.
技术⻛险，需求在技术上难以实现。 性能⻛险，需求实现后，会延⻓系统的响应时间。 安全⻛险，需求实现后，会破坏系统的安全性。 数据库完整性⻛险，需求不容易验证，并且可能导致数据不⼀致性。 开发过程⻛险，需求要求开发⼈员使⽤不熟悉的⾮常规开发⽅法，如形式化规格说明⽅法。 政治⻛险，由于内部政治原因，证实很难实现需求。 法律⻛险，需求可能触犯现⾏法规或者假定了法律的变更。


易变性⻛险，需求很可能在开发过程中不断变化或进化。
17.Why is Change Management needed? Give the name of a tool used in Change Management.
需求是变更的。在开发⽣命周期的任何阶段，需求都有可能变更，可能删除已有需求或者增加新的需求，变更 本身并不会导致困难，但没有管理的变更却会带来麻烦。 开发越往前⾛，需求变更的开销越⼤。 变更可能与⼈为错误有关，但常常是由于内部策略变化或者外部因素⽽引起的。⽆论什么原因，需要强有⼒的 管理政策来建⽴变更请求的⽂档，估计变更的影响，并实现变更。在开发⽣命周期的任何阶段，需求都可能更 改，可能删除已有需求或者增加新的需求。开发越往前⾛，需求变更的开销越⼤。 因为需求变更开销很⼤，每个变更请求必须建⽴⼀个规范化的业务⽤例。 理想的情况下，需求的变更应该由软件配置管理⼯具（software configuration management tool）存储和跟 踪。
18.With the aid of a diagram describe the PCBMER (Presentation, Controller, Bean, Mediator, Entity, and Resource) framework. Each layer should be described in detail.
PCBMER的层
①bean层表示那些预先确定要呈现在⽤户界⾯上的数据类和值对象 ②表示层表示屏幕以及呈现bean对象的UI对象 ③控制器层表示应⽤逻辑 ④实体层响应控制器和中介者 ⑤中介者层建⽴了充当实体类和资源类媒介的通信管道 ⑥资源层负责所有与外部持久数据资源的通信，建⽴和维护与数据源的连接
19.Name and describe the seven main architectural principles.
PCBMER中最重要的体系结构原则：
1、向下依赖原则（DDP）。DDP规定主依赖结构是⾃顶向下的。
2、向上通知原则（UNP）。UNP促进了层与层之间⾃底向上通信的低耦合。 3、相邻通信原则（NCP）。NCP要求每⼀层只能与有直接依赖关系的相邻层通信。 4、显示关联原则（EAP）。EAP表明允许在类之间传递消息。
5、循环去除原则（CEP）。CEP要解决层与层之间的以及层中的类之间的循环依赖。 6、类命名原则（CNP）。CNP原则使得我们通过类名就能了解该类属于哪个层/包。 7、相识包原则（APP）。APP 是 NCP 的推论。相识包由对象在⽅法调⽤参数中传递的接⼝组成，⽽不是由 具体的对象组成。
20.Describe five approaches to discovering classes from requirements.
识别类的⽅法：
1、名词短语⽅法：建议分析员应该阅读需求⽂档中的陈述，从中寻找名词短语。 2、公共类模式⽅法：根据通⽤的对象分类理论来导出候选类。
3、⽤例驱动⽅法：⽤例的图形模型，加上叙述性描述、⾏为和交互模型。
4、CRC⽅法（类-职责-协作者）：CRC⽅法涉及头脑⻛暴式的集体讨论，通过使⽤⼀种特殊制作的卡⽚使其
简单易⾏。
5、混合⽅法：从中间出发，混合以上四种⽅法。
21.Name and describe the four possible semantics for aggregation?
1. ExclusiveOwns聚合
1）依赖性，删除⼀个复合对象，即相关的构建对象都被删除


 2）传递性，如果对象C1是B1的⼀部分，并且B1是A1的⼀部分，那么C1是A1的⼀部分。  3）⾮对称性，如果对象C1是B1的⼀部分，则B1不是C1的⼀部分  4）固定性，如果对象C1是B1的⼀部分，则它绝不是Bi(i≠1)的⼀部分
2. Owns聚合
1）依赖性 2）传递性 3）⾮对称性
3. Has聚合  1）传递性
2）⾮对称性
4. Member聚合
在Member聚合中，⼀个构件对象可以同时属于⼀个以上的复合对象，Member聚合的多重性可以是多 对多的。
22.Describe and contrast Aggregation and Composition.
聚合：表示两个对象之间是整体和部分的弱关系，部分的⽣命周期可以超越整体。如电脑和⿏标。将较弱形式 的聚合简单地称为聚合，并不物理地包含部分对象，具有“通过引⽤”语义。
组合：表示两个对象之间是整体和部分的强关系，部分的⽣命周期不能超越整体，或者说不能脱离整体⽽存在。 组合关系的“部分”，是不能在整体之间进⾏共享的。如⼈和眼睛。将更强形式的聚合成为组合，物理地包含部 分对象，具有“通过值”语义
23.Describe and discuss generalization, including substitutability, polymorphism, abstract classes, abstract operations.
继承是⼀种动作，泛化是⼀种状态。 ⼀个或多个类的公共特性可以抽象到⼀个更⼀般化的类中，这称为泛化。 继承是⼀种泛化，除继承外，泛化的两个⽬的： 可替换性：⼦类对象是超类变量的⼀个合法值。 多态性：同样的操作在不同的类中可以有不同的实现。 多态性在与继承联合使⽤时效果最好。常常在超类中声明⼀个多态操作，但却不提供实现，即给定了操作的型 构（操作名和形式参数列表）。 抽象操作与抽象类不同。抽象类是没有任何直接实例对象的类，带有抽象操作的类就是抽象类。
24. Compare and contrast interfaces and classes.
接⼝除了常量，没有属性、关联或状态。 接⼝有操作，并且所有操作都隐含是公共的和抽象的。 接⼝与类没有关联，但它们可以作为来⾃于类的单项关联的⽬标⽅。 接⼝不等同于抽象类，有抽象操作的类就称为抽象类。
25.Can a class ever model an interface? Explain your answer.
可以，接⼝也是特殊的类,类内部可以有作为“接⼝”的⽅法
26.What does transitivity mean in the context of aggregation?
传递性，若对所有的 a，b，c 属于 X，下述语句保持有效，则集合 X 上的⼆元关系 R 是传递的：「若a 关系 到 b 且 b 关系到 c， 则 a 关系到 c。」
聚合中的传递性指，如果对象A是B的⼀部分，B是C的⼀部分，那么A⼀定是C的⼀部分

  


27.What kinds or relationships are possible between use cases? Describe each. 关联：关联关系建⽴参与者和⽤例之间的通信渠道 包含：允许将被包含⽤例中的公共⾏为分解出来 扩展：通过在特定的扩展点激活另⼀个⽤例来扩展⼀个⽤例的⾏为，从⽽提供⼀种可控的扩展形式 泛化：泛化关系允许⼀个特殊化的⽤例改变基础⽤例的任何⽅⾯
28.Can use cases model concurrency?
⽤例不可表示并发，但是活动图能表示并发
29.Will the state change always occur when the relevant transition to that state has been fired? Explain your answer with an example.
不⼀定。可能指定了⼊⼝动作，那么在状态变化发⽣前，⼊⼝动作需要被满意地完成。
30.What is a UML Stereotype? Using the University Enrollment case study give an example of a stereotype.
在 UML 模型中，构造型是⽤来指出其他模型元素的⽤途的模型元素。UML 提供了⼀组可以应⽤于模型元素的
标准构造型。
可以使⽤构造型来精化模型元素的含义。例如，可以对⼯件应⽤ «library» 构造型以指示它是⼀个特定类型的 ⼯件。可以对使⽤关系应⽤ «global»、«PK»、«include» 构造型，以准确指示⼀个模型元素如何使⽤另⼀个模 型元素。还可以使⽤构造型来描述含义或⽤法不同于另⼀个模型元素的模型元素。 构造型可以具有称为标注定义的属性。将⼀个构造型应⽤于模型元素时，属性的值称为标注值。


31.Explain what a constraint is in the context of a UML class model. Give an example of a class constraint using the University Enrollment case Study.
约束是指条件或限制，是对⼀个元素某些语义的声明，可以⽤⾃然语⾔⽂本或机器可读语⾔来表达。
约束表示附加给被约束元素的额外语义信息。 约束是⼀个断⾔。
例⼦： 在⼤学注册系统中，约束包括：
1、注册系统中将会判断学⽣的年级，给予学⽣相应的选课选择。
2、如果学⽣不及格科⽬的总学分超出限制，则开启留级⽣选课机制
3、系统根据学⽣的选课⽅案，进⾏判断，验证其时间表是否冲突、班级容量是否够⽤等等
32.What is a constrained association? Give an example.
限定关联是在⼆元关联的⼀端，有⼀个属性框 (限定词)，框中包含⼀个或多个属性。这些属性可以作为⼀个索 引码，⽤于穿越从被限定的源类到关联另⼀端的⽬标类的关联。例如， Flight和Passenger之间是多对多的。 然⽽，当类Flight被属性seatNumber和departure限定时，关联的多重性就降为⼀对⼀。由限定词 (flightNumber+seatNumber+departure)引⼊的组合索引码能够被链接到只有零或⼀个Passenger对象上。
33.Describe tags and give an example of their use.

  


34.What are the types of visibility for class attributes and methods? + ：公共可⻅性
 - ：私有可⻅性 # ：保护可⻅性 ~ ：包可⻅性
35.Should you ever use the friend relationship? Explain.
36.Can a subclass access a protected data member or method of a parent class? Give an example. 保护可⻅性应⽤于继承的情景下。如果让基类的私有属性（属性和操作）只能被该类对象所访问，这有时并不 ⽅便。许多情形下，应该允许派⽣类的对象访问基类的其他私有属性。 例⼦：
Person是（⾮抽象）基类，Employee是派⽣类。如果Joe是employee的⼀个对象，那么Joe能够访问Person
的特性。
37.What is derived information? How can it be represented in a UML class diagram? Give an example. 导出信息是⼀种（最经常）应⽤于属性或关联的约束。导出信息从其他模型元素计算得到。严格地说，导出信 息在模型中是冗余的------他可以在需要时计算出来。
导出信息的UML表示法就是在导出属性名或者关联名前加⼀条斜线（/）

  


38.Describe qualified associations and association classes with examples.
关联体现的是两个类之间语义级别的⼀种强依赖关系，⽐如我和我的朋友，这种关系⽐依赖更强、不存在依 赖关系的偶然性、关系也不是临时性的，⼀般是⻓期性的，⽽且双⽅的关系⼀般是平等的。关联可以是单向、 双向的。表现在代码层⾯，为被关联类B以类的属性形式出现在关联类A中，也可能是关联类A引⽤了⼀个类型 为被关联类B的全局变量。在UML类图设计中，关联关系⽤由关联类A指向被关联类B的带箭头实线表示，在关 联的两端可以标注关联双⽅的⻆⾊和多重性标记。
上图中的Job就是⼀个关联类，Person类和Company是存在关系的，但为什么存在关系？是⼯作的缘故。要建 模Person和Company之间的这种⼯作关系，很重要的⼀个内容是⼯作岗位和这⼀岗位的⼯资。如果没有关联 类，那么将⼯资这⼀属性放在Person或是Company类都不合适。
39.Compare and contrast inheritance to encapsulation. ①继承允许⼦类直接访问保护属性，它削弱了封装。 ②当计算涉及不同类的对象时，可能要求这些不同的类彼此是友元或者让元素具有包可⻅性，这就进⼀步破坏 了封装。 ③封装是针对类的概念，不是对象，⽽且在⼤多数的对象程序设计环境中，⼀个对象不能对同⼀个类的另⼀个 对象隐藏任何东⻄

  


40.Explain how interface inheritance can be used to achieve the effect of multiple inheritance in languages such as Java.
⼀，Java不⽀持多继承是由Java的定义决定的，Java最重要的定义，就是因为它是⼀种简单的⾯向对象
解释型的语⾔。
　　⼆，Java不能多重继承是因为这种⽅法很少被使⽤，即使要使⽤也可以通过接⼝来实现多重继承问题。 　　三，Java的定义：
　　1，因为Java: ⼀种简单的，⾯向对象的，分布式的，解释型的（译者注：Java既不是纯解释型也不是纯 编译型的语⾔），健壮的，安全的，架构中⽴的，可移植的，⾼性能的，⽀持多线程的，动态语⾔。 　　2，假设可以多重继承：
　　有两个类B和C继承⾃A；假设B和C都继承了A的⽅法并且进⾏了覆盖，编写了⾃⼰的实现；假设D通过多 重继承继承了B和C，那么D应该继承B和C的重载⽅法，那么它应该继承的是B的还是C的？这就陷⼊了⽭盾， 所以Java不允许多重继承。 但接⼝不存在这样的问题，接⼝全都是抽象⽅法继承谁都⽆所谓，所以接⼝可以继承多个接⼝。
41.Explain three problems that can occur with implementation inheritance. （1）脆弱的基类（fragile base class） 脆弱的基类问题是指在允许对超类的实现进⾏演化的同时，使其⼦类仍然有效并可⽤的问题。如果这种变化还 影响到公共接⼝，情形会进⼀步恶化，如改变⽅法的型构、将⽅法分成两个或多个新⽅法、联合现有的⽅法形 成⼀个更⼤的⽅法。
（2）重载和回调（overriding and callbacks）
实现继承允许对所继承的代码进⾏有选择的重载。⼦类⽅法从其超类中复⽤代码有5种技术： ①⼦类可以继承⽅法实现，并且对实现不做改变。 ②⼦类可以继承代码，在⾃⼰的⽅法中⽤同样的型构包含它（调⽤它）。 ③⼦类可以继承代码，然后⽤⼀个具有相同型构的新的实现来完全重载它。 ④⼦类可以继承空代码（即⽅法声明是空的），然后提供该⽅法的实现。 ⑤⼦类可以只继承⽅法接⼝（即接⼝继承），然后提供该⽅法的实现。
（3）多重实现继承（multiple implementation inheritance） 多重实现继承允许实现⽚段的合并，多重接⼝继承考虑了接⼝协议的合并。多重实现继承并没有给实现继承带 来新的“危害”，它只是放⼤了由脆弱的基类、重载和回调所引发的问题。在需要多重实现继承的地⽅，Java推
荐使⽤多重接⼝继承来提供解决⽅案。
42.Which kind of aggregation needs to be specified with the “frozen” constraint?
The ExclusiveOwns aggregation
43.Explain the difference between synchronous and asynchronous messaging.
同步与异步消息的区别 1、同步消息
 同步消息传递涉及到等待服务器响应消息的客户端。消息可以双向地向两个⽅向流动。本质上，这意味着 同步消息传递是双向通信。即发送⽅向接收⽅发送消息，接收⽅接收此消息并回复发送⽅。发送者在收到接收 者的回复之前不会发送另⼀条消息。通常⽤来传递和数据，实⼼箭头表示。 2、异步消息
 异步消息传递涉及不等待来⾃服务器的消息的客户端。事件⽤于从服务器触发消息。因此，即使客户机被 关闭，消息传递也将成功完成。异步消息传递意味着，它是单向通信的⼀种⽅式，⽽交流的流程是单向的。通 常⽤来传递控制，开放箭头表示。
异步：⽐如A是字符集第⼀个字⺟，唯⼀可⾏的⽅法就是向Z⾛，这意味着是单向通信。 同步：⽐如同步是从字⺟S开始，可能是朝向可能是A或Z，这意味着是双向通信。
44.In a system model can can objects be allowed to communicate in an unrestricted fashion? Explain your answer.

  


可以。
系统模型⼀般不是系统对象本身⽽是现实系统的描述、模仿和抽象。如:地球仪， 是地球原型的本质和特征的 ⼀种近似或集中反映。系统模型是由反映系统本质或特征的主要因素构成的。系统模型集中体现了这些主要因 素之间的关系。
45.In system modeling what benefits do hierarchies have over networks? Explain your answer.
1.各层之间是独⽴的
2.灵活性好
3.结构上可分割开
4.易于实现和维护
5.能促进标准化⼯作
46.What does the term “complexity in the wires” mean?
复杂性具有不同的种类和形态，⼀种简单明了的度量是类之间路径的数量。我们的通信路径定义为类之间存在 的持久或暂时连接。⽤⾏话说，现代企业或电⼦商务的复杂性是“⽤连线数量来度量的”。
47.Explain Structural Complexity. Discuss the structural complexity of hierarchies compared to networks.
分等级的层次组织通过限制类之间的潜在交互路径的数量⽽使复杂度降低了。 通过将类分到层中，只有同⼀层的内部⼀层与层次体系中最下⾯的“相邻”层之间才允许直接的类交互 层之间的通信路径是单向的，层之间任何向上的通信由“依赖最少”的松散耦合来实现的。
48.What is a design pattern? Are design patterns always architectural patterns? Give an example of an architectural pattern. Give three examples of design patterns.
设计模式提供了精化软件系统的元素或它们之间关系的⽅案，它描述通常重复出现的交互设计元素的结构，解 决具有特定环境的⼀般设计问题。当设计模式被⽤于体系结构设计环境时，就可以将其称为体系结构模式。 设计模式分为三种类型，共23种。 创建型模式：单例模式、抽象⼯⼚模式、建造者模式、⼯⼚模式、原型模式。 结构型模式：适配器模式、桥接模式、装饰模式、组合模式、外观模式、享元模式、代理模式。 ⾏为型模式：模版⽅法模式、命令模式、迭代器模式、观察者模式、中介者模式、备忘录模式、解释器模式 （Interpreter模式）、状态模式、策略模式、职责链模式(责任链模式)、访问者模式。 按字典序排列简介如下。
Abstract Factory（抽象⼯⼚模式）：提供⼀个创建⼀系列相关或相互依赖对象的接⼝，⽽⽆需指定它们具体 的类。
Adapter（适配器模式）：将⼀个类的接⼝转换成客户希望的另外⼀个接⼝。Adapter模式使得原本由于接⼝不
兼容⽽不能⼀起⼯作的那些类可以⼀起⼯作。
Bridge（桥接模式）：将抽象部分与它的实现部分分离，使它们都可以独⽴地变化。
体系结构模式(模型 - 视图 - 控制器模式):这种模式也称为MVC模式，将交互式应⽤程序分为三部分，  • 模型 - 包含核⼼功能和数据
• 视图 - 将信息显⽰给⽤户（可以定义多个视图） • 控制器 - 处理来⾃⽤户的输⼊
这样做是为了将信息的内部表⽰与信息呈现给⽤户并从⽤户接受的⽅式分开。 它将组件分离并允许有效的代 码重⽤。
49.What are the benefits to using the Mediator pattern? Use a diagram to explain your answer.

  


Mediator（中介模式）：⽤⼀个中介对象来封装⼀系列的对象交互。中介者使各对象不需要显式地相互引⽤，
从⽽使其耦合松散，⽽且可以独⽴地改变它们之间的交互。
50.Can a components state change? Can a class be implemented by more than one component? 是的，构件没有持久状态，不能与它的拷⻉区分开来 是的，⼀个类可以被多个构件实现，接⼝也可以
51.State the Law of Demeter and the Strong Law of Demeter.
迪⽶特法则（Law of Demeter，LoD）⼜叫最少知识原则（Least Knowledge Principle，LKP），指的是⼀个 对象应当对其他对象有尽可能少的了解。也就是说，⼀个模块或对象应尽量少的与其他实体之间发⽣相互作⽤， 使得系统功能模块相对独⽴，这样当⼀个模块修改时，影响的模块就会越少，扩展起来更加容易。  （2）、关于迪⽶特法则其他的⼀些表述有：只与你直接的朋友们通信；不要跟“陌⽣⼈”说话。  （3）、外观模式（Facade Pattern)和中介者模式（Mediator Pattern）就使⽤了迪⽶特法则。 迪⽶特法则的初衷是降低类之间的耦合，实现类型之间的⾼内聚，低耦合，这样可以解耦。但是凡事都有度， 过分的使⽤迪⽶特原则，会产⽣⼤量这样的中介和传递类，导致系统复杂度变⼤。所以在采⽤迪⽶特法则时要 反复权衡，既做到结构清晰，⼜要⾼内聚低耦合。
Demeter法则说明了在类⽅法中允许什么样的消息⽬标。消息的⽬标只能是下⾯对象之⼀。
1.⽅法的对象本身。
2.⽅法型构中作为参数的⼀个对象。 3.此对象的属性所引⽤的对象。
4.此⽅法创建的对象。
5.全局变量引⽤的对象。


为了限制继承带来的耦合，可以将第3条规则限制在类本身定义的基础上。此类继承来的属性不能被⽤于标识
信息的⽬标对象。将此约束称为Demeter增强法则。
52.Why is mixed-instance cohesion necessary? Give an example using a diagram.
具有混合实例内聚的类具有此类的某些对象没有定义的⼀些特性。此类的某些⽅法仅应⽤于类的对象⼦集，某 些属性也只对⼀部分对象有意义。例如，类Employee可能定义”⼀般”员⼯和管理者对象。管理者有津贴，如果 Employee对象不是管理者，将信息payAllowance()发送给Employee对象就没有意义。
53.In what collaboration model must the roles always be typed? Do collaboration models identify messages? 在复合结构图中，⻆⾊总是被键⼊（有明确定义的类型）。 不，协作模型不标识消息。交互模型标识消息。
54.Explain the three levels of data models.
通常数据库可分为3个层次：外部（概念）数据模型，逻辑数据模型，物理数据模型。 外部模式是指单个应⽤系统所需要的⾼层概念数据模型。最流⾏的概念数据建模技术是实体关系（ER）图。 逻辑模式(有时也称为全局概念模式)是⼀种能够反映系统实现时所采⽤的数据库模型的逻辑存储结构的模型。 逻辑模式是全局集成模型，可以⽀持所有需要从数据库中获取信息的当前应⽤系统和预期应⽤系统。 物理模式专⻔针对特定的DBMS。它定义了数据是如何真正存储在持久存储设备上的，通常指磁盘。物理模式 定义了诸如索引的使⽤和数据的聚类等与有效处理相关的问题。
55.Explain columns, domains and rules. Using a diagram give an example to show how they are used. 列：属性 域：列可以取值的合法集 规则：对列和域进⾏约束
56.What is referential integrity? How is it useful for the mapping from a UML class model? Explain the four declarative referential integrity constraints.


（1）—参照完整性：⼜称引⽤完整性，指表间的规则，作⽤于有关联的两个或两个以上的表，通过使⽤主键 和外键（或唯⼀键）之间的关系，使表中的键值在相关表中保持⼀致。参照完整性约束⽤来维护表间关系。 （2）四个参考完整性约束
①Upd(R)；Del(R)：限制更新或删除操作 ②Upd(C)；Del(C)：级联操作
③Upd(N)；Del(N)：设置为空值 ④Upd(D)；Del(D)：设置为省缺值
57.What is a trigger? How is it related to referential integrity? Can a stored procedure call a trigger? Explain.
（1）触发器是⼀个⼩程序，⽤扩展的SQL语句编写，当定义了触发器的表发⽣修改操作时⾃动执⾏（触
发）。
（2）规则和描述性参照完整性约束允许在数据库中定义简单的业务规则，但⽤来定义更复杂的规则或定义规 则的异常，那就不够了。RDB对这个问题的解决⽅案就是触发器，触发器可以⽤来实施适⽤于数据库的所有业 务规则和更复杂的参照完整性约束。
（3）触发器是⼀种不能被调⽤的特殊存储过程，当对⼀个表执⾏insert、update或delete操作时⾃我触发。
58.What mathematical concept is a relational database model based on? The set theory（ 集合理论）
59.Referring to the following figure, consider a situation in which an entity object needs to be unloaded to the database as a result of an update operation, instead of delete. How would the sequence diagram differ in such a case?
删除⼀个对象先删除对应数据库内的相应数据，再销毁现有的对象
59.Describe briefly the locks in pessimistic concurrency control.
事务要处理的每个持久对象都需要设置锁，有4种对象锁：
1.排他（写）锁：必须等到持有这种锁的事务完成并释放该锁之后，才能处理其他事务。


2.更新（预写）锁：其他事务可以读取对象，但只要有需要，就可以将持有这种锁的事务升级为排他模式。
3.读（共享）锁：其他事务可以读取对象并且可能得到这个对象的更新锁。
4.⽆锁：其他事务可以随时更新对象，因此它只适⽤于允许脏读的应⽤系统，即⼀个事务读取的数据在该事务 完成之前可能已经修改或者甚⾄删除了。
60.What is a compensating transaction? How can it be used in program design?
事务补偿即在事务链中的任何⼀个正向事务操作，都必须存在⼀个完全符合回滚规则的可逆事务。 可以在数据库的内部做事务处理，不过⽤得最⼴的还是transaction绑定到具体的connection上。 补偿型的例⼦，在⼀个⻓事务（ long-running ）中 ，⼀个由两台服务器⼀起参与的事务，服务器A发起事务， 服务器B参与事务，B的事务需要⼈⼯参与，所以处理时间可能很⻓。如果按照ACID的原则，要保持事务的隔 离性、⼀致性，服务器A中发起的事务中使⽤到的事务资源将会被锁定，不允许其他应⽤访问到事务过程中的 中间结果，直到整个事务被提交或者回滚。这就造成事务A中的资源被⻓时间锁定，系统的可⽤性将不可接 受。WS-BusinessActivity提供了⼀种基于补偿的long-running的事务处理模型。还是上⾯的例⼦，服务器A的 事务如果执⾏顺利，那么事务A就先⾏提交，如果事务B也执⾏顺利，则事务B也提交，整个事务就算完成。但 是如果事务B执⾏失败，事务B本身回滚，这时事务A已经被提交，所以需要执⾏⼀个补偿操作，将已经提交的 事务A执⾏的操作作反操作，恢复到未执⾏前事务A的状态。
61.Describe the five levels of SQL programming interfaces.
第⼀层：设计⼈员/DBA <- SQL数据定义语⾔ 第⼆层：特别⽤户/DBA <- SQL数据操纵语⾔ 第三层：程序员 <- SQL嵌⼊式语⾔ 第四层：设计⼈员/程序员 <- 4GL/SQL（应⽤⽣成器） 第五层：设计⼈员/程序员 <- 过程化的SQL（存储过程）
62.Can the amount of database recovery time be controlled by the designer/DBA? Explain.
可以，DBA可以通过设置检查点出现的频率来控制恢复时间的⻓短。

  


63.Describe briefly the levels of transaction isolation.
1.未提交读(脏读)：事务t1修改了⼀个对象，但是还没有提交，这时事务t2读取了这个对象，如果t1回滚这个事 务，则t2获得了⼀个数据库中根本不存在的对象。
2.提交读(⾮重复读)：t1读取了⼀个对象，⽽t2修改了这个对象，t1再次读取这个对象，但这次获得了同⼀个对 象的不同值。
3.重复读 (虚读)：t1读取了⼀组对象，⽽t2在这组对象中插⼊⼀个新对象，这时t1再次读取这组对象，则会看
到⼀个“虚拟的”对象。
4.序列化(可重复读)：t1和t2可以并发执⾏，即使交替执⾏这两个事务所产⽣的结果都是⼀样的，就像它们是⼀ 次执⾏⼀个事务⼀样（这可称为可序列化的执⾏）。
64.Explain what a long transaction is and what they are used for.
占⽤整个逻辑⽇志空间在⼀定⽐例以上的事务，就叫做“⻓事务”。“⻓事务”意味着可能由于跨越过多的⽇志⽂ 件，导致需要循环使⽤的⽇志⽂件不能及时释放。从⽽造成数据库系统挂起⽆法正常⼯作的可能性。 ⻓事务的作⽤主要是，为了保护数据库的良好运⾏，防⽌由于事务占⽤过多的⽇志空间，在系统没有进⾏跟踪 的情况下发⽣失效时不允许⾃动回滚。