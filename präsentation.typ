#let title-page(title: [], subtitle: [], faculty, body) = {
  // set page( numbering: "1")
  set page(
    margin: (rest: 1.5in, left: 2.7in),
    paper: "presentation-16-9",
    fill: color.linear-rgb(255, 252, 225),
  )
  set text(font: "EB Garamond", size: 16pt, lang: "de")
  set heading(numbering: "1.1.1", supplement: metadata("Shown"))
  line(start: (0%, 0%), end: (8in, 0%), stroke: (thickness: 2pt))

  align(horizon + left)[
    #text(size: 32pt, title)\
    #v(1em)
    #subtitle
  ]

  align(bottom + left)[
    #link("mailto: marius.niveri@studium.uni-hamburg.de")[Marius Niveri (7436282)]\
    #datetime.today().display("[day].[month].[year]")
    #align(
      right,
      link(faculty)[Universität Hamburg],
    )
  ]
  pagebreak()
  set text(font: "EB Garamond", size: 20pt, lang: "de")
  set page(
    margin: auto,
    footer: context [
      #text(
        size: 9pt,
        grid(
          columns: (1fr, 1fr, 1fr),
          align(
            left,
            [
              #title \
              #link("mailto: marius.niveri@studium.uni-hamburg.de")[Marius Niveri (7436282)]
            ],
          ),
          align(
            center,
            counter(page).display("1 von 1", both: true),
          ),
          align(
            right,
            link(faculty)[Universität Hamburg],
          ),
        ),
      )
    ],
  )
  align(center, text(outline(indent: auto, title: none), size: 15pt))
  pagebreak()
  body
  pagebreak()
  set text(font: "EB Garamond", size: 16pt, lang: "de")
  heading(
    "Literaturverzeichnis",
    numbering: none,
    level: 1,
    supplement: metadata("Hidden"),
  )
  bibliography("bib.yaml", title: none)
  v(1em)
  text("Der Quellcode dieser Präsentation ist auf GitHub verfügbar: ")
  link("https://github.com/m4r1vs/presentation-ai-hardware")[github.com/m4r1vs/presentation-ai-hardware]
}

#show heading.where(level: 1, supplement: metadata("Shown")): it => {
  block(width: 100%)[
    #set align(horizon + center)
    #set text(30pt, weight: "bold")
    #v(32%)
    #text(it.body)
  ]
  pagebreak()
}

#show: body => title-page(
  title: [Hardware-Beschleunigung für ML/AI: GPUs und TPUs],
  subtitle: [
    Seminar Supercomputer: Forschung und Innovation\
    Bei
    #link("mailto: anna.fuchs@uni-hamburg.de")[Anna Fuchs]
    und
    #link("mailto: jannek.squar@uni-hamburg.de")[Jannek Squar]\
  ],
  "https://wr.informatik.uni-hamburg.de/start",
  body,
)


#show figure: it => {
  it.body
  block[
    #set text(size: 12pt) // Adjust the size to be smaller
    #text(it.caption)
  ]
}

#show cite: it => {
  super(it)
}

= Einführung

== Aktueller Stand
#v(16pt)

- Markt für KI-Chips aktuell stark von hochparallelisierbaren\
  Prozessoren dominiert: @NvTheChipBBC
  - GPUs, TPUs, NPUs, etc.
- Microsoft kaufte 2024 eine halbe Millionen NVIDIA Karten (H100, H200, ..) @MicrAcq500Mio
  - Insgesamt über $30$ Milliarden Euro an Kosten.
#v(3em)
#align(
  right,
  [
    #sym.arrow.r.double Warum sind xPUs so gut geeignet für KI- und ML- Anwendungen?
  ],
)

#pagebreak()

= Berechnung von KI/ML

== Feedforward Neural Network <ff-nn>

- Gewichte werden im Training angepasst.
- Beispiel: $(P_n, D_n)$ Bilder $P$ und deren Beschreibungen $D$ als $n$ Trainingsdaten.
  - $b$ Batches: $n/b$ große Partitionen der Daten
  - $e$ Epochen: Rückpropagation wird $e$-Mal auf den Daten ausgeführt.

  #sym.arrow.r.double Berechnungen der Batches kann parallel ausgeführt werden. \
  #sym.arrow.r.double Nach jeder Epoche werden Gewichte aggregiert.

- Alternativ falls Modell zu groß für Speicher:

  #sym.arrow.r.double Aufteilung der Ebenen auf Prozessoren.

#v(1em)
#align(
  right,
  [
    #sym.arrow.r.double Berechnungen fast ausschließlich Vektor-Matrix Multiplikationen!
  ],
)

#pagebreak()

== Recurrant Neural Networks

- Schlecht zu parallelisieren, da Gradienten voneinander abhängig sind:
#columns(
  2,
  [
    #figure(
      image("Hopfield-net-vector.svg", height: 65%),
      caption: [RNN mit 4 Neuronen @HopRNNSVG],
    )
    #block(fill: color.linear-rgb(236, 226, 160), inset: 8pt, radius: 4pt)[
      Allerdings könnte sich das bald ändern! @ParallelizingNonLinearSeqModels
    ]
  ],
)

#pagebreak()

== Transformer und Generatoren

- 2017 wurde "Attention Is All You Need" von Google veröffentlicht.
- Grundstein für *parallelisierbare* Transformer wurde daduch gelegt. @AttnIsAllYouNeed \
  - Jahr darauf Veröffentlichung von GPT-1 (Generative Pretrained Transformer 1). @GPT1GitHub

#pagebreak()

=== Simplifizierte Funktionsweise

#align(center)[_"Ich esse den Hamburger"_]

1. Eingabe in Vektoren ("Tokens") aufteilen.
  - z.B. _"Ich"_ $= (231, 231, 534, 4, 321, 342, ...)$
2. Vektorwerte durch $mono("Attention")$ aktuallisieren.
  - Sind wir oder #text("🍔", size: 14pt) mit _"Hamburger"_ gemeint?
3. Kontext von Tokens durch Feed-Forward#super(ref(<ff-nn>)) erweitern.
  - z.B. Fakten speichern. @FactFind
4. Schritt 2 und 3 bis zur gewünschten Genauigkeit wiederholen.
#sym.arrow.r.double *Ergebnis* (mit viel mehr Trainingsdaten): \
- Inhaltlich ähnliche Tokens liegen räumlich dicht beieinander.
- Statistisch bestimmbar, welches nächste Token am wahrscheinlichsten ist.

#pagebreak()

=== "Aufmerksamkeit" Berechnen
- Beziehungen zwischen Wörtern wird durch $mono("Attention")$ berechnet:

#let unimp(x) = text(fill: color.linear-rgb(0, 0, 0, 116), $#x$)

$
  mono("Attention")(Q,K,V) := mono("softmax")((Q K^unimp(T)) / unimp(sqrt(d_k)))V
$

// Queries: Fragen (z.B. Adjektive vor mir?)
// Keys: Antworten (z.B. Ja, hier!)
// Values: Werte   (z.B. Änderungen in der "Farb" Dimension)
#h(1em)Matrizen $Q, K, V$: Queries, Keys, Values \ \
- Gibt an, wie relevant ein Token für die Aktualisierung eines anderen Tokens ist.
- Kann sehr gut parallelisiert werden ($mono("MultiheadAttention")$).
#v(2em)
#align(
  right,
  [
    #sym.arrow.r.double Berechnungen fast ausschließlich Vektor-Matrix Multiplikationen!
  ],
)

#pagebreak()

= Matrix Multiplikationen Parallel Ausführen

== Kernel Definieren

*Erinnerung*: Gegeben $arrow(v) = mat(v_1;v_2)$ und $M = mat(M_(1,1), M_(1,2); M_(2,1), M_(2,2))$,
gilt: $arrow(v) dot M = mat(v_1 dot M_(1,1) +v_2 dot M_(2,1);v_1 dot M_(1,2) + v_2 dot M_(2,2))$

#show raw.where(block: true): block.with(
  fill: color.linear-rgb(235, 204, 205),
  inset: 8pt,
  radius: 4pt,
)

#align(center)[
  #text(size: 18pt)[
    ```C
    __global__ void vectorMatrixMulKernel(float *V, float *M, float *RESULT, int n) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < n) {
            float value = 0.0f;
            for (int k = 0; k < n; k++) {
                value += V[k] * M[k * n + row];
            }
            RESULT[row] = value;
        }
    }
    ```]
]

#pagebreak()

== Kernel Ausführen

#align(center)[
  #text(size: 16pt)[
    ```C
    void vecMatMul(float *V, float *M, float *RESULT, int n) {
      float *d_V, *d_M, *d_RESULT;

      cudaMalloc(&d_V, n * sizeof(float));
      // .. gleiches für M und RESULT

      cudaMemcpy(d_V, V, n * sizeof(float), cudaMemcpyHostToDevice);
      // .. gleiches für M

      int blockSize = 256; // Je nach GPU Modell und Matrix Größe
      int gridSize = (n + blockSize - 1) / blockSize;

      vectorMatrixMulKernel<<<gridSize, blockSize>>>(d_V, d_M, d_RESULT, n);

      cudaMemcpy(C, d_C, sizeVector, cudaMemcpyDeviceToHost);
      // TODO: Speicher freigeben mit cudaFree();
    };
    ```]
]

#pagebreak()

== Was passiert in der Hardware?

#figure(
  image("./cpu-gpu-mem.png", height: 80%),
  caption: [CPU und GPU Abbildung],
)


#pagebreak()

= Zusammenfassung

== Vergleich von "Matrix Proccessing Units"

=== TPU
- TPU #sym.arrow.l.r.double "Tensor Processing Unit"
  - Stark von Google geprägter Begriff.
  - Leicht energieeffizienter als GPUs der gleichen Generation.@TPUvsGPU

=== NPU
- NPU #sym.arrow.l.r.double "Neural Processing Unit"
  - Nicht klar definierter Begriff.
  - Meist werden Chips in mobilen Endgeräten bezeichnet, die ML-Modelle ausführen sollen.
    - z.B. "Neural Engine" von Apple für "Apple Intelligence".@NeuralEngineAAPL
  - Manchmal auch $text("TPU"), text("GPU") in text("NPU")$

#pagebreak()

== Brauchen wir GPUs im HPC?
#v(2em)
*Ja*, weil... \
#h(1em) ... GPUs sind sehr gut geeignet für maschinelles Lernen und KI. Ein Trend der bleibt. \
#h(1em) ... Durch gekonnte Programmierung können auch viele andere Programme \
#h(2em) durch GPUs beschleunigt werden.
