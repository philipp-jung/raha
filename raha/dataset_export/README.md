# Ensembling Experimente
Hi Phillip, hier die besprochenen Datensätze. Die Datensätze entstehen bei der Reinigung
zweier Tabellen: `137` und `1481`. In `metadata_137.json` und `metadata_1481.json`
findest du die Parameter pro Datensatz aufgeführt:

- `labeled_tuples` ist die Anzahl an User-Inputs, also Zeilen der Ground-Truth,
aus denen Reinigungsfeatures erstellt werden.
- `column` gibt die Spalte an, die gereinigt wird.

## Ensembling Datensätze
Du erhältst sechs `.parquet` Dateien pro Spalte, `labeled_tuples` und Datensatz:
- `x_train, y_train` sind die Trainingsdaten, mit denen das Ensemblingmodell
trainiert wird. Die Daten stammen aus den User-Inputs.
- `x_test` sind die Features, die aus den zu reinigenden Zeilen erzeugt werden.
- `y_test` ist die Ground-Truth für die Reinigung. `y_test` liegt im Programm
selbst nicht vor, ich habe es für deine Experimente rausgeschrieben -- sich an
`y_test` zu bedienen bedeutet, bei der Reinigung zu schummeln.
- `synth_x_train, synth_y_train` sind Trainingsdaten, die aus der zu reinigenden
Tabelle erzeugt werden.

## Die Tabellen selbst
Ich habe dir noch zusätzlich die zu reinigenden Tabellen eingefügt,
`1481_clean.parquet, 1481_dirty.parquet, 137_clean.parquet, 137_dirty.parquet`.
In den Datensätzen sind missing values, gekennzeichnet als leere Strings,
komplett zufällig in 5% aller Zellen eingefügt.

Falls du noch Fragen hast, melde dich bitte bei mir! Viel Erfolg bei den Experimenten :)
