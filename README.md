# SylloBio-NLI: Testing AI's ability to think logically in biomedical research

## The Question

While there is a lot of potential for applications of AI assistant tools in biomedical research, the reliability of their underlying technology (large language models - LLMs) and their ability to logically compose their responses are crucial factors in adopting the technology into this field. This leads to a fundamental question:  Can current AI models accurately and consistently draw logical conclusions from biomedical information?

In everyday life we often use simple logic: “If it rains, the ground gets wet.” That’s a basic syllogism: a rule that lets us draw a conclusion from two facts. In medicine, doctors, researchers and AI tools need to do the same thing, but with far more complex facts: gene‑pathway relationships, drug mechanisms, disease pathways, etc. If an AI cannot reliably follow these logical rules, it might give wrong answers or miss crucial connections, which could be dangerous in a clinical setting.

This is incredibly important for several reasons. Biomedicine is full of complex relationships: how genes interact, how drugs affect the body, how diseases progress. If we want AI to help with tasks like drug discovery, personalized medicine, or even just understanding scientific papers, it needs to be able to reason logically. A wrong conclusion could have serious consequences, leading to ineffective treatments, misdiagnosis, or wasted research efforts. We wanted to see if the AI we’re increasingly trusting for medical decisions is ready for these critical tasks, testing whether it is actually thinking the right way, not just guessing.

## Methodology

We created a large dataset called SylloBio-NLI. This dataset contains a series of logical puzzles based on real biomedical information. These puzzles are designed to test different types of reasoning, using 28 different logical structures (like "If A then B", or "Either A or B is true"). <br />

We then tested eight different AI models on this dataset, using two main approaches:

- **Zero-shot learning**: We asked the AI to solve the puzzles without giving it any specific examples. This tests the AI’s inherent reasoning ability.
- **Few-shot learning**: We gave the AI a few examples of solved puzzles before asking it to solve new ones. This tests whether the AI can learn from a small amount of guidance.


By analyzing how well the AI models performed on these puzzles, we could assess their reasoning capabilities:

- For each model and each logical pattern, we recorded how often the model answered correctly, how precise it was, and whether it could pick out the necessary facts from a list of many facts (a harder “premise‑selection” task).
- We also shuffled the wording (negations, complex phrases, etc.) to see if the model’s answers were robust to surface changes.

The following diagram illustrates the complete research approach: generation of syllogistic arguments from domain-specific ontologies, parameterized input to LLMs, and evaluation tasks including textual inference and premise selection.</p>

<img src="syllobio_diag.png" style="width: 60%;">

## Key Findings

<table style="width: 90%">
    <colgroup>
        <col style="width: 36%"/>
        <col style="width: 64%"/>
    </colgroup>
    <thead>
    <tr class="header">
        <th><strong>What was tested</strong></th>
        <th><strong>Result</strong></th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
        <td><strong>Zero‑shot performance</strong></td>
        <td>Most models did <em>not</em> beat random guessing (50%) on the simple “if‑then” syllogisms. Only one model (Gemma‑7B‑it) reached about 64% accuracy.
        </td>
    </tr>
    <tr class="even">
        <td><strong>Few‑shot help</strong></td>
        <td>Adding a few example syllogisms improved some models dramatically (e.g., Meta‑Llama‑3‑8B’s accuracy jumped 43%). But the boost was uneven, with some models performing worse with examples.
        </td>
    </tr>
    <tr class="odd">
        <td><strong>Scheme‑specific differences</strong></td>
        <td>Models were best on “generalised modus ponens” (the classic “if‑then” rule) and struggled most with “disjunctive syllogism” (the “either‑or” rule).
        </td>
    </tr>
    <tr class="even">
        <td><strong>Sensitivity to wording</strong></td>
        <td>When the same logical structure was phrased differently (e.g., using negations or more complex
            verbs), model accuracy dropped sharply, especially in zero‑shot mode. Few‑shot examples helped but didn’t eliminate the problem.
        </td>
    </tr>
    <tr class="odd">
        <td><strong>Distractors &amp; factuality</strong></td>
        <td>Adding irrelevant facts (distractors) hurt performance for some models, while changing real gene names to made‑up ones had little effect, suggesting the models rely more on structure than on biomedical knowledge.
        </td>
    </tr>
    </tbody>
</table>

**Bottom line:**

 - Current AI models can <em>sometimes</em> reason logically in a medical context, but they are fragile. 
 - They often ignore the actual medical facts and instead latch onto surface patterns.
 - Few‑shot prompting can help, but it’s not a universal fix. The improvement depends on the model’s architecture and training background

## Conclusion

Despite the progress in LLM capabilities, the SylloBio-NLI study underscores the challenges associated with applying these models to complex reasoning tasks in the biomedical domain.  Further research is needed to develop more robust and reliable methods for leveraging LLMs in biomedical applications, focusing on improving logical comprehension and mitigating the reliance on superficial patterns.

## Resources

- <a href="https://aclanthology.org/2025.naacl-long.371/"><b>Research paper</b></a>
- <a href="https://github.com/neuro-symbolic-ai/SylloBio-NLI"><b>Code repository</b></a>

**Citation:**

<pre>
@inproceedings{wysocka-etal-2025-syllobio,
    title = "{S}yllo{B}io-{NLI}: Evaluating Large Language Models on Biomedical Syllogistic Reasoning",
    author = "Wysocka, Magdalena  and
              Carvalho, Danilo  and
              Wysocki, Oskar  and
              Valentino, Marco  and
              Freitas, Andre",
    editor = "Chiruzzo, Luis  and
              Ritter, Alan  and
              Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics:
                 Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.371/",
    doi = "10.18653/v1/2025.naacl-long.371",
    pages = "7235--7258",
    ISBN = "979-8-89176-189-6"
}
</pre>