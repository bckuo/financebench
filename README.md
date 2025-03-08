# Apply FinanceBench on GPT-4o-mini and Claude 3.5 Haiku

This project evaluate the performance of GPT-4o-mini and Claude 3.5 Haiku on open-book financial question answering using FinanceBench.

### Experiment setting

||Claude 3.5 Haiku|GPT-4o mini|
|-|-|-|
|Closed Book|all|all|
|In Context|group 0|group 0|
|In Context (Reverse)|group 0|all|
|Oracle|group 0|group 0|
|Oracle (Reverse)|group 0|group 0|
|Shared Store|group 0|group 0|
|Single Store|group 0|group 0|

Temperature (0.01) and max tokens (2048) are both same with FinanceBench setting.

Due to the time and usage constraint, espeacially the price of Claude 3.5, currently only group 0 (30 QAs) had all setting conducted, and only 20 of them are annotated.

P.S. Since there was only one annotator for this project and his financial knowlege was not extensive, the annotated result should be used with caution.

### Observation
<p align="left">
    <img src="img/claude-3-5-haiku-20241022.png" alt="drawing" style="width: 400px; display: block; margin: 0 auto; text-align:center;"/>
</p> <p align="right">
    <img src="img/gpt-4o-mini.png" alt="drawing" style="width: 400px; display: block; margin: 0 auto; text-align:center;"/>
</p>

There are 10 metrics-generated questions, 5 domain-relevant questions, and 5 novel-generated questions in the annotated set. 
<br>

The result is same as expectation, that Oracle setting with the unrealistic highest accuracy, In Context with the second highest, VectorDB the third, and Close Book the last.

### Other

1. I recorded the number of characters cut off from the regulating function in the In Context setup and found they are quite high. This make me wonder if the information that LLM need to answering question is also being cut off. LLM also often fail to find information with In Context setup. Thus, this could be the next study subject after finished testing and annotating whole dateset.

2. Data Error: The question financebench_id_00438 in data should use total revenue in justification, but subscription is recoreded instead.

### Reference
@misc{islam2023financebenchnewbenchmarkfinancial,
      title={FinanceBench: A New Benchmark for Financial Question Answering}, 
      author={Pranab Islam and Anand Kannappan and Douwe Kiela and Rebecca Qian and Nino Scherrer and Bertie Vidgen},
      year={2023},
      eprint={2311.11944},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2311.11944}, 
}