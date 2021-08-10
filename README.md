# Automated Hate Speech Detection and the Problem of Offensive Language
Repository for Thomas Davidson, Dana Warmsley, Michael Macy, and Ingmar Weber. 2017. "Automated Hate Speech Detection and the Problem of Offensive Language." ICWSM. You read the paper [here](https://aaai.org/ocs/index.php/ICWSM/ICWSM17/paper/view/15665).


# NOTE: This repository is no longer actively maintained. Please do not post issues regarding the compatibility of the existing code with new versions of Python or the packages used. I will not accept any pull requests. If you plan to use this data or code in your research, please review the [issues](https://github.com/t-davidson/hate-speech-and-offensive-language/issues), as several Github users have suggested changes or improvements to the codebase.

## 2019 NEWS
We have a new paper on racial bias in this dataset and others, you can read it [here](https://arxiv.org/abs/1905.12516)


***WARNING: The data, lexicons, and notebooks all contain content that is racist, sexist, homophobic, and offensive in many other ways.***

You can find our labeled data in the `data` directory. We have included them as a pickle file (Python 2.7) and as a CSV. You will also find a notebook in the `src` directory containing Python 2.7 code to replicate our analyses in the paper and a lexicon in the `lexicons` directory that we generated to try to more accurately classify hate speech. The `classifier` directory contains a script, instructions, and the necessary files to run our classifier on new data, a test case is provided.


***Please cite our paper in any published work that uses any of these resources.***
~~~
@inproceedings{hateoffensive,
  title = {Automated Hate Speech Detection and the Problem of Offensive Language},
  author = {Davidson, Thomas and Warmsley, Dana and Macy, Michael and Weber, Ingmar}, 
  booktitle = {Proceedings of the 11th International AAAI Conference on Web and Social Media},
  series = {ICWSM '17},
  year = {2017},
  location = {Montreal, Canada},
  pages = {512-515}
  }
~~~

***Contact***
We would also appreciate it if you could fill out this short [form](https://docs.google.com/forms/d/e/1FAIpQLSdrPNlfVBlqxun2tivzAtsZaOoPC5YYMocn-xscCgeRakLXHg/viewform?usp=pp_url&entry.1506871634&entry.147453066&entry.1390333885&entry.516829772) if you are interested in using our data so we can keep track of how these data are used and get in contact with researchers working on similar problems.

If you have any questions please contact `thomas dot davidson at rutgers  dot edu`.
