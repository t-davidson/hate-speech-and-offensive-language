# Automated Hate Speech Detection and the Problem of Offensive Language
Repository for Thomas Davidson, Dana Warmsley, Michael Macy, and Ingmar Weber. 2017. "Automated Hate Speech Detection and the Problem of Offensive Language." ICWSM. You read the paper [here](https://aaai.org/ocs/index.php/ICWSM/ICWSM17/paper/view/15665).

***WARNING: The data, lexicons, and notebooks all contain content that is racist, sexist, homophobic, and offensive in many other ways.***

You can find our labeled data in the `data` directory. We have included them as a pickle file (Python 2.7) and as a CSV. You will also find a notebook in the `src` directory containing Python 2.7 code to replicate our analyses in the paper and a lexicon in the `lexicons` directory that we generated to try to more accurately classify hate speech. The `classifier` directory contains a script, instructions, and the necessary files to run our classifier on new data, a test case is provided.

***Please cite our paper in any published work that uses any of these resources.***
~~~
@inproceedings{hateoffensive,
  title={Automated Hate Speech Detection and the Problem of Offensive Language},
  author={Davidson, Thomas and Warmsley, Dana and Macy, Michael and Weber, Ingmar}, 
  booktitle={Proceedings of the 11th International AAAI Conference on Web and Social Media},
  series={ICWSM '17},
  year={2017},
  location = {Montreal, Canada}
  }
~~~

We are also presenting another paper based on this work at the Association for Computational Linguistics 1st Workshop on Abusive Language in August 2017, you can find the pre-print [here](https://arxiv.org/abs/1705.09899).

We would also appreciate it if you could fill out this short [form](https://docs.google.com/forms/d/e/1FAIpQLSdrPNlfVBlqxun2tivzAtsZaOoPC5YYMocn-xscCgeRakLXHg/viewform?usp=pp_url&entry.1506871634&entry.147453066&entry.1390333885&entry.516829772) if you are interested in using our data so we can keep track of how these data are used and get in contact with researchers working on similar problems.

If you have any questions please contact `trd54@cornell.edu`.
