#ifndef STEMMER_H
#define STEMMER_H

#include <QString>

class Stemmer
{
public:
    static QString stemWord(const QString &word);
    static QStringList stemSentence(const QString &sentence);
};

#endif // STEMMER_H
