1. Was ist euer Thema im Kontext von HPC?

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

2. Was ist die Herausforderung/Problemstellung bei diesem Thema?

Es gibt sehr viele verschiedene GPUs, von denen manche besser als
andere für KI geeignet sind. Die Herausforderung ist es, zu wissen,
für welchen Einsatzbereich welche Hardware die beste ist.

Außerdem wird im HPC Bereich nicht nur Production Code ausgeführt sondern
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
sind schnell veraltet. Es kann sein

3. Was ist die Idee/Lösungsvorschlag/Kernkonzept?
4. Welche offenen Punkte/Fragestellungen gibt es noch?
