# Gossip算法的研究与实现(Multi-thread using Java)

[TOC]

---

## Chapter I. Gossip算法介绍

#### 1.算法背景

​		由于卡夫卡集群的特性，在系统运行一段时间后（默认配置是7天），会自动清除掉过期的记录，因此每个周期之后加入的节点都会丢失一部分数据。于是，我们需要一个机制能不依赖卡夫卡集群来实现数据的一致性。

​		这就是 Gossip算法。当卡夫卡集群无法保证数据一致性时，通过此算法，保证系统最终数据一致。同时，还可支持节点间各种类型的消息传播。

#### 2.Gossip算法概述

##### 2.1 算法简介

> Gossip, or anti-entropy,  is an attractive way of replicating state that does not have strong consistency requirements

​		顾名思义，类似于流言传播的概念，Gossip是一种可以按照自己的期望自行选择与之交换信息的节点的通信方式

​		在一个有界网络中，每个节点都随机地与其他节点通信，经过一番杂乱无章的通信，最终所有节点的状态都会达成一致。每个节点可能知道所有其他节点，也可能仅知道几个邻居节点，只要这些节可以通过网络连通，最终他们的状态都是一致的，当然这也是疫情传播的特点。

​		Gossip是一种**去中心化**、**容错**而又**最终一致性**的绝妙算法，其收敛性不但得到证明还具有**指数级**的收敛速度。使用 Gossip 的系统可以很容易的将 Server扩展到更多的节点，满足弹性扩展轻而易举。

​		Gossip常见于大规模、无中心的网络系统，可以用于众多能接受“最终一致性”的领域：失败检测、路由同步、Pub/Sub、动态负载均衡。

##### 2.2 算法特点

​		Gossip不要求节点知道所有其他节点，因此具有**去中心化**的特点，节点之间完全对等，不需要任何的中心节点。

​		Gossip算法又被称为反熵（Anti-Entropy），熵是物理学上的一个概念，代表杂乱无章，而反熵就是在杂乱无章中寻求一致，这充分说明了Gossip的特点：

> 在一个有界网络中，每个节点都随机地与其他节点通信，经过一番杂乱无章的通信，最终所有节点的状态都会达成一致。每个节点可能知道所有其他节点，也可能仅知道几个邻节点，只要这些节可以通过网络连通，最终他们的状态都是一致的。

​		Gossip算法是一个最终一致性算法，其无法保证在某个时刻所有节点状态一致，但可以保证在”最终“所有节点一致，**”最终“是一个现实中存在，但理论上无法证明的时间点。**

##### 2.3 Gossip协议满足的条件

- 协议的核心包括周期性，成对性，内部进程交互
- 交互期间的信息量大小固定
- 节点交互后，至少一个agent获知另一个agent的状态
- 通信不可靠
- 交流的频率远远低于消息的传输延迟
- 对端选择的随机性，或者从全集，或者从部分集合
- 由于副本的存在，传输的信息具有隐式冗余

##### 2.4 协调机制

​		协调机制是讨论在每次2个节点通信时，如何交换数据能达到最快的一致性，也即消除两个节点的不一致性。
​		协调所面临的最大问题是，因为受限于网络负载，不可能每次都把一个节点上的数据发送给另外一个节点，也即每个Gossip的消息大小都有上限。在有限的空间上有效率地交换所有的消息是协调要解决的主要问题。

> “Efficient Reconciliation and Flow Control for Anti-Entropy Protocols”中描述了两种同步机制
> **1）precise reconciliation**
>
> > ​		precise reconciliation希望在每次通信周期内都非常准确地消除双方的不一致性，具体表现为相互发送对方需要更新的数据，因为每个节点都在并发与多个节点通信，理论上很难做到。precise reconciliation需要给每个数据项独立地维护自己的version，在每次交互是把所有的(key,value,version)发送到目标进行比对，从而找出双方不同之处从而更新。但因为Gossip消息存在大小限制，因此每次选择发送哪些数据就成了问题。当然可以随机选择一部分数据，也可确定性的选择数据。对确定性的选择而言，可以有最老优先（根据版本）和最新优先两种，最老优先会优先更新版本最新的数据，而最新更新正好相反，这样会造成老数据始终得不到机会更新，也即饥饿。
>
> **2）Scuttlebutt Reconciliation**
>
> > ​		Scuttlebutt Reconciliation 与precise reconciliation不同之处是，Scuttlebutt Reconciliation不是为每个数据都维护单独的版本号，而是为每个节点上的宿主数据维护统一的version。比如节点P会为(p1,p2,...)维护一个一致的全局version，相当于把所有的宿主数据看作一个整体，当与其他节点进行比较时，只需比较这些宿主数据的最高version，如果最高version相同说明这部分数据全部一致，否则再进行precise reconciliation。

##### 2.5 Merkle tree

​		信息同步无疑是gossip的核心，Merkle tree(MT)是一个非常适合同步的数据结构。
​		简单来说 Merkle tree就是一颗hash树，在这棵树中，叶子节点的值是一些hash值、非叶节点的值均是由其子节点值计算hash得来的，这样，一旦某个文件被修改，修改时间的信息就会迅速传播到根目录。需要同步的系统只需要不断查询跟节点的hash，一旦有变化，顺着树状结构就能够在 logN 级别的时间找到发生变化的内容，马上同步。
​		在Dynamo中，每个节点保存一个范围内的key值，不同节点间存在有相互交迭的key值范围。在去熵操作中，考虑的仅仅是某两个节点间共有的key值范围。MT的叶子节点即是这个共有的key值范围内每个key的hash，通过叶子节点的hash自底向上便可以构建出一颗MT。Dynamo首先比对MT根处的hash，如果一致则表示两者完全一致，否则将其子节点交换并继续比较的过程。

##### 2.6 时间复杂度 logN 的证明

![1570799966173](Gossip算法的研究与实现(multi-thread).assets/1570799966173.png)

![1570799985330](Gossip算法的研究与实现(multi-thread).assets/1570799985330.png)

##### 2.7 算法具体描述
###### 2.7.1 gossip 协议的类型

前面说了节点会将信息传播到整个网络中，那么节点在什么情况下发起信息交换？这就涉及到 gossip 协议的类型。目前主要有两种方法：

Anti-Entropy（反熵）：以固定的概率传播所有的数据
Rumor-Mongering（谣言传播）：仅传播新到达的数据

###### 2.7.2 Anti-Entropy

Anti-Entropy 的主要工作方式是：每个节点周期性地随机选择其他节点，然后通过互相交换自己的所有数据来消除两者之间的差异。Anti-Entropy 这种方法非常可靠，但是每次节点两两交换自己的所有数据会带来非常大的通信负担，以此不会频繁使用。

Anti-Entropy 使用“simple epidemics”的方式，所以其包含两种状态：susceptible 和 infective，这种模型也称为 SI model。处于 infective 状态的节点代表其有数据更新，并且会将这个数据分享给其他节点；处于 susceptible 状态的节点代表其并没有收到来自其他节点的更新。

###### 2.7.3 Rumor-Mongering

Rumor-Mongering 的主要工作方式是：当一个节点有了新的信息后，这个节点变成活跃状态，并周期性地联系其他节点向其发送新信息。直到所有的节点都知道该新信息。因为节点之间只是交换新信息，所有大大减少了通信的负担。

Rumor-Mongering 使用“complex epidemics”方法，相比 Anti-Entropy 多了一种状态：removed，这种模型也称为 SIR model。处于 removed 状态的节点说明其已经接收到来自其他节点的更新，但是其并不会将这个更新分享给其他节点。

因为 Rumor 消息会在某个时间标记为 removed，然后不会发送给其他节点，所以 Rumor-Mongering 类型的 gossip 协议有极小概率使得更新不会达到所有节点。

一般来说，为了在通信代价和可靠性之间取得折中，需要将这两种方法结合使用。

###### 2.7.4 gossip 协议的通讯方式

不管是 Anti-Entropy 还是 Rumor-Mongering 都涉及到节点间的数据交互方式，节点间的交互方式主要有三种：Push、Pull 以及 Push&Pull。

Push：发起信息交换的节点 A 随机选择联系节点 B，并向其发送自己的信息，节点 B 在收到信息后更新比自己新的数据，一般拥有新信息的节点才会作为发起节点。
Pull：发起信息交换的节点 A 随机选择联系节点 B，并从对方获取信息。一般无新信息的节点才会作为发起节点。
Push&Pull：发起信息交换的节点 A 向选择的节点 B 发送信息，同时从对方获取数据，用于更新自己的本地数据。

## Chapter II. 作业描述与实现

#### 1.作业描述

##### 1.1 简述

使用Gossiping协议实现去中心化的平均数算法（使用任意语言均可）



> 假设初始所有GossipNode线程的Message为0，为Passive即不包含任何消息的节点，而在第一轮迭代开始时一个GossipNode的Messge为10，为Active，则该Node将会随机选择其他节点进行信息传播即感染，而进行选择的Node与被选择的Node会消除两者之间的信息差异，即进行Message取平均数，被感染的Node与初始Message为10的Node的Message都变为5，此后获得了信息的Node即为Active可以在每轮循环中随机选择其他（**包括已经被隔离的**)节点进行感染，直到他兴趣值不断降低被确认为DeadNode被Kill也就是隔离为止 ，**隔离**仅仅意味着该节点无法主动选取其他节点进行信息传播



##### 1.2 定义

| Definition | Meaning                                                      |
| :--------- | ------------------------------------------------------------ |
| 误差       | 整个算法的核心在于携带初始信息的节点在Gossip之后使得其他节点通过传播均取得相同的信息，假设 InitialActiveNode.getMessage() = 10, 而numberOfNodes = 10,则初始信息的平均值与理想状况下Gossip结束后的Message平均值应该都为1。在实际算法运行中，可能最后节点携带的信息值为1.05或0.98一类的，此时用 理想的平均值 1 减去 其信息值 , 求差值之后将所有节点的差值求平均值，即为所求算法误差的平均值 |
| 收敛轮数   | 整个Gossip从开始到终止迭代的轮数                             |
| K 值       | K是用来计算兴趣值的影响因子                                  |
| 节点隔离   | **隔离**仅仅意味着该节点无法主动选取其他节点进行信息传播，被隔离的节点仍然可以被其他ActiveNode选择为信息传播的目标。具体实现方法为 在每轮gossip开始之前，所有独立的ActiveNode生成一个随机数verdictNum(裁决数字)，取值范围在（0,1）之内,对每个Active的节点进行判定，若该节点的valueOfInterest < verdictNum，则该节点被隔离 (一次生成一个verdictNum对所有节点进行相同条件的判定，sounds fair, 但是并不适用于分布式系统) |
| 兴趣值     | 获得了信息的Node即为Active可以在每轮循环中随机选择其他节点进行感染，在每轮GossipNode开始感染之前，其成功传播的概率为valueOfInterest，每轮开始传播之前进行一次判定，也可以认为valueOfInterest为该节点存活的概率，若判定无法传播也即没有兴趣，会被确立为死节点并被清除, valueOfInterest *= 1/k  ,k为常值,他选择的有可能为Message为0的PassiveNode也有可能为Message不为0的其他ActiveNode,而当一个Message = c 的 ActiveNode与另一个同样Message = c的Node进行交换时，由于信息相同而受挫，每次受挫都会导致该Node的兴趣值降低，即为 |

$$
valueOfInterest = 1/(k)^n   【n为受挫的次数】
$$

可以理解为 valueOfInterest 为每轮Gossip开始时决定该节点是否继续存活保持Active而不被隔离的几率

##### 1.3 结果要求

| Requirements ( 难度 =  Level 5 ) |
| -------------------------------- |
| 1.误差与K值之间的关系图          |
| 2.误差与节点个数之间的关系图     |
| 3.收敛轮数与K值之间的关系图      |
| 4.收敛轮数与节点个数之间的关系图 |
| 5.多线程模拟多节点代码           |
| 6.输入输出结果展示               |



#### 2.代码实现

##### 1.GossipNode

```JAVA

package com.company;

import java.util.Objects;
import java.util.Random;

/**
 * A class representing a gossip Node.
 *
 * @author Vellichor
 * ID: 20175045
 * Name: 高歌
 */
class GossipNode {

    /**
     * message ---- 信息
     * downTimes ---------------受挫次数
     * valueOfInterest ------- 兴趣值
     * rounds-----------轮数
     * status ---------- 1 - Active      0- Passive      2- Dead
     * id ---------- 节点 ID 在生成 ArrayList时特定设立为与 Thread 线程与 节点索引相同的数字便于识别
     */

    private volatile Double message;
    private volatile int downTimes;
    private volatile Double valueOfInterest;
    private int rounds;
    private volatile int status;
    private int id;


    GossipNode(Double message, int downTimes, Double valueOfInterest, int rounds, int status, int id) {
        this.message = message;
        this.downTimes = downTimes;
        this.valueOfInterest = valueOfInterest;
        this.rounds = rounds;
        this.status = status;
        this.id = id;
    }

    /**
     * return a random number which is not equal to another unwanted number
     *
     * @param unwantedNum
     * @param numberRange
     * @return
     */
    private static int getRandomNum(int unwantedNum, int numberRange) {
        Random random = new Random();
        int randomNum = random.nextInt(numberRange) + 1;//用于生成该节点在ArrayList中的索引
        if (randomNum == unwantedNum) {
            getRandomNum(unwantedNum, numberRange);
        }
        return randomNum;
    }

    /**
     * 随机生成并返回一个[0.1)的 Double数值 用于判断节点是否被隔离
     *
     * @return
     */
    private static Double generatorDouble0To1() {

        return new Random().nextDouble();
    }

    /**
     * 用于求 两个节点交换信息后的平均信息值
     *
     * @param Message_1
     * @param Message_2
     * @return
     */
    private static Double getAvgMessage(Double Message_1, Double Message_2) {
        return (Message_1 + Message_2) / 2;
    }

    /**
     * @param targetNode      当前节点所选择感染的目标节点
     * @param value_K         K值  用于影响兴趣值的递减
     * @param minusToleration 对于判断节点携带信息是否相同时所用的误差限度（因为 Double 为高精度数字）
     * @return 1--节点继续保持活跃  2----节点被隔离
     * （默认 status = 0 时节点为未携带信息时的 Passive 状态）
     */
    public int startGossiping(GossipNode targetNode, int value_K, Double minusToleration) {

        //每轮遍历ArrayList找寻存活的Active节点进行一轮的感染
        System.out.println();
        System.out.println("当前节点  Message = " + message + " down" +
                "Times =  " + downTimes + "兴趣值valueOfInterest = " + valueOfInterest + "状态Status = " + status);
        System.out.println("生成（0，1）内的随机数 对该节点进行判定");
        Double verdictNum = generatorDouble0To1();
        System.out.println("生成随机数为 " + verdictNum);

        if (valueOfInterest >= verdictNum) {
            System.out.println("兴趣值上界 >= 判定数字  在兴趣值区间内，节点继续保持活跃");
            //若节点为活跃状态，开始寻找被感染者
            //轮数加1
            rounds++;
            System.out.println("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 轮数:  " + rounds + " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
            System.out.println("------------------节点 " + id + " 号开始寻找被感染者---------------- ");

            //若二者信息相同(由于为高精度Double，故考虑二者绝对值差值在0.05之内就为相同， ActiveNode节点受挫， 受挫次数 downTimes+1
            double minusValue = targetNode.message - message;
            if (Math.abs(minusValue) < minusToleration) {
                downTimes++;
                //根据改变的受挫次数设置 新的兴趣值
                valueOfInterest = message / value_K;

                System.out.println("二者携带信息相同（信息差值在允许范围内）： currentNodeMessage: " + message + " targetNodeMessage = " + targetNode.message );
                System.out.println("感染者受挫，" + "downTimes = " + downTimes + "兴趣值valueOfInterest = " + valueOfInterest);
            }else{
                //二者信息 均被设置为 其信息的平均值
                Double avgMessage = getAvgMessage(targetNode.message, message);
                targetNode.setMessage(avgMessage);
                message = avgMessage;
                System.out.println("交换信息后 二者信息均变为 avgMessage = " + avgMessage);
            }

            switch(targetNode.status) {
                case 0:
                    targetNode.setStatus(1);//该节点状态被设置为活跃
                    System.out.println("节点被感染 Status设置为 1 = Active");
                    targetNode.setRounds(rounds);//被感染节点初次被感染，继承当前节点的轮数
                    break;
                case 1:
                    System.out.println("节点之前已经被感染 Status为 1 = Active 保持不变");
                case 2:
                    System.out.println("被感染者已经被隔离, status 不变 仍然为 2 = Dead ");
            }

            return 1;//节点继续保持活跃

        } else {
            System.out.println("兴趣值上界 < 判定数字    判定数字不在兴趣值区间内，节点被隔离, 轮数不会增加");
            targetNode.setStatus(2);

            return 2;//节点被隔离
        }
    }
	// 部分代码省略
}

```



##### 2.initGossip

```JAVA
package com.company;

import java.util.ArrayList;
import java.util.concurrent.Semaphore;

public class initGossip {

    private int num;

    public initGossip(int num) {
        this.num = num;
    }

    /**
     * 初始化节点列表
     * message ---- 信息  Default----0.0
     * downTimes ---------------受挫次数----Default------0
     * valueOfInterest ------- 兴趣值--Default----1.0
     * rounds-----------轮数--Default------0
     * status ---------- 1 - Active      0- Passive      2- Dead----Default------0
     * id ---------- 节点 ID 在生成 ArrayList时特定设立为与 Thread 线程与 节点索引相同的数字便于识别
     */
    static ArrayList<GossipNode> initGossipNodeList(int num){
        ArrayList<GossipNode> nodeArrayList = new ArrayList<>(num);
        for (int i = 0; i < num; i++) {
            // 首先将所有节点设置为 Default 条件
            GossipNode defaultNode = new GossipNode(0.0, 0, 1.0, 0, 0, i);
            nodeArrayList.add(defaultNode);
        }
        // 将第一个节点设置为 Status=1--Active & Message=1.0
        nodeArrayList.get(0).setStatus(1);
        nodeArrayList.get(0).setMessage(1.0);
        System.out.println("节点初始化全部完成，索引与ID皆为0的 节点被设置为 Message = 1.0 的初始感染节点");
        return nodeArrayList;
    }

    // 初始化信号量都为1个
    static ArrayList<Semaphore> initSemaphoreList(int num) {
        ArrayList<Semaphore> semaphoreArrayList = new ArrayList<>(num);
        for (int i = 0; i < num; i++) {
            semaphoreArrayList.add(new Semaphore(1));
        }
        System.out.println("信号量初始化完成");
        return semaphoreArrayList;
    }
}

```

##### 3.GossipingMultiThread

```JAVA
package com.company;

import java.io.*;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

public class GossipingMultiThread {

    /**
     * value_K : K值 决定兴趣值递减程度
     * numOfNodes : 节点数  同时也为线程数
     * minusToleration : 判断 Message 是否相同时的误差容忍度
     */
    private static GossipSettings  settings = new GossipSettings(3,20,0.01);
    private static int numOfNodes = settings.getNumberOfnodes();
    private static int value_K = settings.getValue_K();
    private static Double minusToleration = settings.getMinusToleration();//判断Message是否相同时候的误差容忍度
    //调用 initGossip中的方法，在建立ArrayList时已经完成了 ArrayList中数据的初始化
    private static ArrayList<GossipNode> nodeArrayList = initGossip.initGossipNodeList(numOfNodes);
    private static ArrayList<Semaphore> semaphoreArrayList = initGossip.initSemaphoreList(numOfNodes);

    //使用ArrayList控制线程池中线程的关闭
    private static ArrayList<Integer> threadPoolStartList = new ArrayList<Integer>(numOfNodes);
    private static ArrayList<Integer> threadPoolEndList = new ArrayList<Integer>(numOfNodes);

    public static void main(String[] args) throws IOException {

        //将控制台Console中的输出全部打印到 consoleOutput.txt 中
        PrintStream oldPrintStream = System.out;
        FileOutputStream bos = new FileOutputStream("consoleOutput.txt");
        MultiOutputStream multi = new MultiOutputStream(new PrintStream(bos),oldPrintStream);
        System.setOut(new PrintStream(multi));

        for (int i=0;i<numOfNodes;i++){
            threadPoolStartList.add(0);
            threadPoolEndList.add(0);
        }

        //用于判断线程池的
        // 开线程池，每个线程搭载一个节点
        ExecutorService ex = Executors.newCachedThreadPool();
        ex.execute(new ActiveMessageThread(0));
        // 关闭线程池的条件
        int allEqual = 0;
//        while (shutDownSign == 0 ){
        while (allEqual == 0 && !threadPoolEndList.isEmpty() ){
            allEqual = 1;
            for (int i=0;i<numOfNodes;i++){
                if (threadPoolStartList.get(i).equals(threadPoolEndList.get(i))){
                    allEqual = 1;

                }else {
                    allEqual = 0;
                }
            }
            //如果记录线程开关的两个 ArrayList 对应索引位置的数值都相同 前 第一个启动的线程已经结束
            //终止线程池
//            if (threadPoolEndList.get(0) == 1 && allEqual ==1){
//                shutDownSign = 1;
//                System.out.println("如果记录线程开关的两个 ArrayList 对应索引位置的数值都相同 而且 第一个启动的线程已经结束 ");
//                System.out.println("终止线程池");
//            }

        }
        ex.shutdown();
        // 确认线程池运行完毕

//        //控制台输出可能受线程池关闭时间影响不准,准确输出以该文件为准
//        writeResult();
        // 写日志
        CsvWriter.writeResultToCSV(nodeArrayList,value_K);

        for (int i = 0;i<numOfNodes;i++){
            System.out.println(nodeArrayList.get(i).getRounds());
        }


    }
    // 线程类
    private static class ActiveMessageThread implements Runnable {
        private int currentIndex;// 该线程搭载的信息发送节点编号

        private ActiveMessageThread(int currentIndex) {
            this.currentIndex = currentIndex;// 初始化时为线程指定所搭载的节点编号
        }

        public void run() {
            while (true) {
                //序号为currentIndex的线程开始运行
                //threadPoolStartList中索引为currentIndex的数值设为1
                threadPoolStartList.set(currentIndex,1);

                int gossipResult = 1;// 初始化是否传递成功
                int minResource = 0;// 初始化较小编号资源的编号
                int maxResource = 0;// 初始化较大编号资源的编号
                try {
                    // 随机选择要传达的节点,不能是自己
                    int targetIndex;
                    do {
                        targetIndex = (int) (Math.random() * numOfNodes);
                    } while (targetIndex == currentIndex);
                    // 进行资源排序，防止死锁
                    if (currentIndex > targetIndex) {
                        maxResource = currentIndex;
                        minResource = targetIndex;
                    } else {
                        maxResource = targetIndex;
                        minResource = currentIndex;
                    }
                    // 先获取小编号信号量
                    semaphoreArrayList.get(minResource).acquire();
                    // 才能获取大编号信号量
                    semaphoreArrayList.get(maxResource).acquire();
                    int targetStatus = nodeArrayList.get(targetIndex).getStatus();

                    // 通过startGossiping发送消息并且获取 当前节点状态
                    //1---------节点继续保持活跃
                    //2----------节点被隔离
                    // gossipResult 表示这轮感染成功开始  targetStatus == 0 表示目标节点尚未被感染 可以开启新线程
                    gossipResult = nodeArrayList.get(currentIndex).startGossiping(nodeArrayList.get(targetIndex), value_K, minusToleration);
                    if (gossipResult == 1 && targetStatus == 0) {
                        // 如果感染了新的节点,就启动搭载他的线程,
                        // 这里如果感染者和易感者的数本来就相等,线程也会启动,因为相当于感染者把感染全部节点的需求传达给了易感者
                        new Thread(new ActiveMessageThread(targetIndex)).start();

                    }
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    // 集体释放信号量
                    semaphoreArrayList.get(minResource).release();
                    semaphoreArrayList.get(maxResource).release();
                }
                // gossipResult = 2 代表节点已经被隔离
                if (gossipResult == 2) {// 终止线程

                    //序号为currentIndex的线程开始运行
                    //threadPoolEndList中索引为currentIndex的数值设为1

                    threadPoolEndList.set(currentIndex,1);
                    break;
                }
            }
        }
    }
}

```

#### 3.输入输出结果展示

###### 3.1 Console控制台输出展示

![1571926475885](Gossip算法的研究与实现(multi-thread).assets/1571926475885.png)

###### 3.2 结果数据打印

![1571926503527](Gossip算法的研究与实现(multi-thread).assets/1571926503527.png)

#### 4.结果分析

![1571924466380](Gossip算法的研究与实现(multi-thread).assets/1571924466380.png)

![1571924632284](Gossip算法的研究与实现(multi-thread).assets/1571924632284.png)

![1571924730193](Gossip算法的研究与实现(multi-thread).assets/1571924730193.png)



![1571924850345](Gossip算法的研究与实现(multi-thread).assets/1571924850345.png)

![1571925811832](Gossip算法的研究与实现(multi-thread).assets/1571925811832.png)



