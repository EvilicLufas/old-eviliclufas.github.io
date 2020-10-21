# 6th Week Homework

####  ID : 20175045

> TextBook, pp 242, Exercises 1.  Answer in English within 200 words.
>
> > Discuss the underlying conceptual model for the following activities: using an ATM; buying a plane ticket on the Web; using a public information kiosk; setting a VCR to record a programme.
>
> 

Conceptual modeling is about describing the semantics of software applications at a high level of abstraction. Specifically, conceptual modelers (1) describe structure models in terms of entities, relationships, and constraints; (2) describe behavior or functional models in terms of states, transitions among states, and actions performed in states and transitions; and (3) describe interactions and user interfaces in terms of messages sent and received and information exchanged. In their typical usage, conceptual-model diagrams are high-level abstractions that enable clients and analysts to understand one another, enable analysts to communicate successfully with application programmers, and in some cases automatically generate (parts of) the software application.

#### Activity 1 : using an ATM



<img src="6th Week Homework.assets/1586180781193.png" alt="1586180781193" style="zoom: 50%;" />

<img src="6th Week Homework.assets/1586180980773.png" alt="1586180980773" style="zoom: 50%;" />



The functional specifications are documented graphically in Dataflow Diagrams (DFDs) with a brief description below.
**STEP 0:** (Defining the scope of the existing system under study) - This accomplished by drawing the context diagram for the existing system.
**STEP 1:** (Documentation of how the existing system work) This was accomplished by drawing the Physical Data Flow Diagrams. These DFDs specify the current implementation of existing system, and would answer questions such as: • Who performs the tasks in existing system?
• How they are performed in existing system?
• When or how often they are performed in existing system?
• How the data is stored in existing system?
• The task process, is it secured in existing system?
• How is the data flows implemented in existing system?
These physical DFDs may be leveled, or, if the system is not very large, prepared all on a single DFD.
**STEP 2:** (Documentation of what the existing system does.)This is documented in logical DFDs of existing system. Deriving these logical DFDs of an existing system from the physical DFDs involve abstraction of all implementation details as shown in the AFIM later in this section. From the formal design description, all the implementation details are abstracted from the user hence OOADM applied logically. These logical DFDs are leveled in order to reduce the perceived complexity of the system, and balanced in order to assure consistency in the design.
**STEP 3:** (Documentation of what the proposed system will do.) After step 2, an examination or performance evaluation is carried out to ascertain why the existing system (ATM) does not meet the user requirements, and how it can be modified in order to meet such needs. The result is a set of logical DFDs which describe what the modified (proposed AFIM) system will do. These functional specifications are devoid of implementation considerations, and therefore rather abstract specifications of the proposed system. These logical DFDs are also levelled and balanced.
**STEP 4:** (Documentation of how the proposed system will work.) The logical DFDs of the proposed AFIM system derived in step 3 above are then examined to determine which implementation of it meets the user requirements most efficiently. The result is a set of physical DFDs of the proposed system. They answer questions such as: • Who will perform the various tasks in AFIM?
• How they will be performed in AFIM?
• When or how often they will be performed in AFIM?
• How the data will be stored in AFIM?
• The task process, is it secured in AFIM?
• How the data flows will be implemented in AFIM?
In this step, man-machine boundaries are drawn, and media selected for all data flows and data stores. 

<img src="6th Week Homework.assets/1586181144882.png" alt="1586181144882" style="zoom: 33%;" />

<img src="6th Week Homework.assets/1586181179890.png" alt="1586181179890" style="zoom: 50%;" />



<img src="6th Week Homework.assets/1586181266579.png" alt="1586181266579" style="zoom: 50%;" />

<img src="6th Week Homework.assets/1586181356058.png" alt="1586181356058" style="zoom: 50%;" />

#### Activity 2 : buying a plane ticket on the Web

<img src="6th Week Homework.assets/loop.jpg" alt="查看源图像" style="zoom: 50%;" />

#### Activity 3 : using a public information kiosk

<img src="6th Week Homework.assets/1586172277548.png" alt="1586172277548" style="zoom:50%;" />

<img src="6th Week Homework.assets/1586172193602.png" alt="1586172193602" style="zoom:50%;" />

<img src="6th Week Homework.assets/Kiosk-Sketches.jpg" alt="查看源图像" style="zoom: 67%;" />



#### Activity 4 : setting a VCR to record a programme

<img src="6th Week Homework.assets/1586181465353.png" alt="1586181465353" style="zoom: 50%;" />

<img src="6th Week Homework.assets/esa_concept_fig_3.gif" alt="查看源图像" style="zoom: 67%;" />

![查看源图像](https://tse1-mm.cn.bing.net/th?id=OIP.BHdVgeSLcyVCAXsXFJ1PLgHaEs&pid=Api&rs=1)

![查看源图像](https://web.esrc.unimelb.edu.au/ICAD/objects/images/D00000056.jpg)

![查看源图像](http://www.cisco.com/c/dam/en/us/td/i/200001-300000/220001-230000/224001-225000/224669.eps/_jcr_content/renditions/224669.jpg)