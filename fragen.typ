#let title-page(title: [], subtitle: [], fill: yellow, body) = {
  // set page( numbering: "1")
  set page(margin: (top: 1.5in, rest: 2in))
  set text(font: "Ubuntu", size: 10pt, lang: "de", top-edge: 1em)
  set heading(numbering: "1.1.1")
  line(start: (0%, 0%), end: (8.5in, 0%), stroke: (thickness: 2pt))
  align(horizon + left)[
    #text(size: 21pt, title)\
    #v(1em)
    #subtitle
  ]

  align(bottom + left)[
    #link("mailto: marius.niveri@studium.uni-hamburg.de")[Marius Niveri]\
    #datetime(year: 2025, month: 04, day: 01).display()
  ]
  pagebreak()
  set page(
    fill: none,
    margin: auto,
    footer: context [
      #text(
        size: 8pt,
        [
          #title\
          #link("mailto: marius.niveri@studium.uni-hamburg.de")[Marius Niveri]
          #h(1fr)
          #counter(page).display(
            "1",
            both: false,
          )
        ],
      )
    ],
  )
  body
}

#show: body => title-page(
  title: [Fragestellungen zu _Hardware-Beschleunigung für ML/AI: GPUs und TPUs_],
  subtitle: [
    Seminar Supercomputer: Forschung und Innovation\
    Bei
    #link("mailto: anna.fuchs@uni-hamburg.de")[Anna Fuchs]
    und
    #link("mailto: jannek.squar@uni-hamburg.de")[Jannek Squar]
  ],
  body,
)

#set par(spacing: 2.2em, justify: false)

= Was ist euer Thema im Kontext von HPC?
\
Ich habe mich mit Hardware Beschleunigung von KI und ML Algorithmen
beschäftigt. GPUs sind sowohl für den Hobby-Einsatz auch als den
Profibereich begehrter denn je. Nvidida ist die oder eine der
wertvollsten Börsennotierten Firmen der Welt und die neuen GPU Modelle
sind in Sekunden für den doppelten Preis auf Ebay zu haben.

Alle paar Wochen gibt es eine neue Schlagzeile, dass ein weiteres Unternehmen
oder eine weitere Regierung Milliarden in KI investiert. Das Geld
landet dann oft direkt im Bau von neuen Datencentern und
der Anschaffung von GPUs.

Interssiert hat mich, warum GPUs im Vergleich zu CPUs so gut geeignet
sind für die Anwendung von KI. Dafür habe ich versucht zu verstehen,
welche Berechnungen für die Ausführung von KI durchgeführt werden
müssen und warum diese Berechnungen gut auf GPUs laufen.

Dafür habe ich mir einerseits konkrete KI-Modelle (FeedForward-Neural Network,
Recurrent Network, Transformer) und andererseits GPU Architekturen angeschaut.

Da KI im HPC Bereich direkt eingesetzt wird, indem sie auf den HPC-Knoten
angewandt wird, ist die richtige Wahl der Hardware kritisch.

= Was ist die Herausforderung/Problemstellung bei diesem Thema?
\
Es gibt sehr viele verschiedene GPUs, von denen manche besser als
andere für KI geeignet sind. Die Herausforderung ist es, zu wissen,
für welchen Einsatzbereich welche Hardware die beste ist.

Außerdem wird im HPC Bereich nicht nur _Production_ Code ausgeführt sondern
auch geforscht und entwickelt. Das bedeutet, dass ein besonders
tiefes Verständnis der Hardware nötig ist, um Fehler beim Debugging
helfen zu können und die Software besser auf die Hardware optimieren zu können.

In der Welt der GPUs gibt es mit NVidia und AMD zwei wesentliche Player wobei
manchmal auch Nichen wie z.B. ARM Ampere die richtige Wahl sein können.

Um Kunden und Forschenden das beste Software Angebot aufstellen zu können,
ist es optimal, wenn die Software nicht auf CUDA oder ROCm beschränkt ist.
Also, wenn sowohl NVidia, als auch AMD Karten verfügbar sind.

Für gewisse (meist öffentlich staatliche) Projekte könnte die Open-Source
Eigenschaft von ROCm sogar notwendig sein.

Da die Welle der KI sich immernoch so schnell bewegt und nicht kleiner
zu werden scheint, gibt es im Internet auch viel Fluff und Informationen
sind schnell veraltet.

Meiner Meinung nach ist der massive Fortschritt in der KI im wesentlich durch
zwei Dinge getrieben: bessere Hardware und größere Trainingsdatensätze (Internet hat sich etabliert).

#pagebreak()
= Was ist die Idee/Lösungsvorschlag/Kernkonzept?
\
Die GPU kann man am ehesten als Miniatur-Version eines Supercomputers betrachten. Es gibt
die sogenannten SMs (Streaming Multiprocessors), die auf den gemeinsamen VRAM zugreifen können
und von der CPU ihre Aufgaben erhalten. Die SMs wiederum sind bisschen wie eine besondere Art von Mehrkern CPU.
Sie bestehen aus eigenen Kernen (Kernel), die allerdings nur eine Art von Berechnung gleichzeitig durchführen können und
auf den Speicher des SMs zugreifen.

Hat man also eine Aufgabe, die gut parallelisiert werden kann wobei die Verarbeitung keinen stetig gemeinsamen
Speicher voraussetzt und die einzelnen Berechnungen von der Mathematik her gleich sind,
so kann diese Aufgabe sehr gut von einer GPU ausgeführt werden.

Das Paradebeispiel hier ist die Matrixmultiplikation -- sowohl herkömmlich wo die Zellen meist Pixel einer Grafik waren
als auch in der KI, wo es vieldimensionale Vektorräume sind, die generell Informationen enkodieren.

Die Idee ist nun häufig, die Nachfrage nach KI einfach mit mehr Hardware zu beantworten. Durch
die Knappheit und Kosten von GPUs werden Modelle aber mittlerweile auch immer mehr auf Effizienz optimiert.

Dass das durchaus möglich ist, hat z.B. das R1 LLM von DeepSeek gezeigt. Aber auch die "destillierten" Modelle vieler
anderer Hersteller, die ein Bruchteil der ursprünglichen Anforderungen an die Hardware haben aber dennoch ähnlich gute Ergebnisse liefern bzw.
auf bestimmte Einsatzgebiete optimiert sind.

= Welche offenen Punkte/Fragestellungen gibt es noch?
\
Auch wenn GPUs immer beliebter sind, quasi pflicht im modernen Datencenter, gibt es auch immer mehr Ideen für noch optimiertere Hardware für
KI-Anwendung. So spielen FPGAs (Programmierbare Mikrocontroller) durchaus eine Rolle und könnten Teil der Lösung sein, wenn man
den Energieverbrauch der GPUs als Problem betrachtet.

Auch ist noch sehr offen, ob die Zukunft der alltäglich angewandten KI in der Cloud oder lokal in der eigenen Hardware ist.
Immer mehr Geräte werden als "KI-Ready" vermarktet - gerade mit Hinsicht auf bessere Privatsphäre - wobei die nützlichen Modelle
dann doch in Datencentern laufen und ihre Antworten über das Internet streamen.

Da KI Modelle meist einen relativ hohen Speicherbedarf haben, kann die Idee "Unified Memory" die richtige sein. Hier teilt sich
die CPU ihren Speicher dynamisch mit anderen Chips, was die Kosten der Hardware senkt und so einen höheren GPU Speicher erlaubt.
So können auch die besten Modelle mit hohem Speicherbedarf lokal genutzt werden.

In der Forschung wird die Ansteuerung der Hardware auch weiterhin diskutiert werden. In fünf Jahren kann gut eine neue
Python Bibliothek der Standard für KI-Ansteuerung sein. Die Chancen stehen dann auch gut, dass CUDA nicht als first-class citizen
behandelt wird sondern AMD, etc. Karten auch ohne Umwege nutzbar sind.

Solange KI die heutige Bedeutung hat, wird die Frage nach der richtigen Hardware relevant sein. Da KI Modelle meist eh schon abichtlich eine gewisse unschärfe ("Temperatur")
eingebaut haben, könnten sogar Quantencomputer eingesetzt werden, die theoretisch mehrere Berechnungen parallel auf einem Kern durchführen können.
