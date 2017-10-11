#include <algorithm>
#include <QStringList>
#include "Porter2Stemmer/porter2_stemmer.h"
#include "stemmer.h"

using namespace std;
using namespace Porter2Stemmer;

QString Stemmer::stemWord(const QString &word)
{
    string s = word.toStdString();
    stem(s);
    return QString::fromStdString(s);
}

static bool isLetter(const char &c)
{
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
        return true;
    else
        return false;
}

static string replacePuncts(const string &str)
{
    string ret = str;
    replace_if(ret.begin() , ret.end(), [] (const char &c) { return !isLetter(c); }, ' ');
    return ret;
}

QStringList Stemmer::stemSentence(const QString &sentence)
{
    QString copy(sentence.toLower());
    copy = copy.remove("'s").remove("`s").replace("'t", "t").replace("`t", "t");
    copy = QString::fromStdString(replacePuncts(copy.toStdString()));

    QStringList words = copy.split(" ", QString::SkipEmptyParts);
    QStringList ret;
    foreach (QString word, words)
    {
        QString stemmed = stemWord(word);
        if (stemmed.size() > 1)
            ret.append(stemmed);
    }
    return ret;
}
