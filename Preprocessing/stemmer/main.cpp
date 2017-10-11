#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QtGlobal>
#include <QDebug>
#include "stemmer.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    if (argc != 3)
    {
        QString exeName= a.applicationFilePath().split("/").last();
        qDebug(QString("Usage: %1 Input_Path Output_Path").arg(exeName).toLatin1());
        return -1;
    }

    QString input(argv[1]);
    QString output(argv[2]);
#ifdef Q_OS_WIN32
    if (input.size() > 2)
    {
        if (input[1] == ':' && input[2] == '\\')
            input = input.replace("\\", "/");
        else
            input = QFileInfo(QString("%1/%2").arg(QDir::currentPath()).arg(input)).absoluteFilePath();
    }
    if (output.size() > 2)
    {
        if (output[1] == ':' && output[2] == '\\')
            output = output.replace("\\", "/");
        else
            output = QFileInfo(QString("%1/%2").arg(QDir::currentPath()).arg(output)).absoluteFilePath();
    }
#else
    if (input.startsWith("/"))
        input = QFileInfo(input).absoluteFilePath();
    else
        input = QFileInfo(QString("%1/%2").arg(QDir::currentPath()).arg(output)).absoluteFilePath();

    if (output.startsWith("/"))
        output = QFileInfo(output).absoluteFilePath();
    else
        output = QFileInfo(QString("%1/%2").arg(QDir::currentPath()).arg(output)).absoluteFilePath();
#endif

    if (!QFile::exists(input))
    {
        qDebug(QString("Cannot find %1").arg(QDir::toNativeSeparators(input)).toLatin1());
    }

    QFile *inp = new QFile(input, 0);
    inp->open(QFile::ReadOnly);

    QFile *outp = new QFile(output, 0);
    outp->open(QFile::WriteOnly);

    while (!inp->atEnd())
    {
        QString line = QString::fromUtf8(inp->readLine()).remove(QRegExp("[\r\n]"));
        if (line.isEmpty())
            continue;

        QStringList contents = line.split("\t", QString::KeepEmptyParts);
        if (contents.size() != 4)
        {
            qDebug("The input file must have 4 columns.");
            continue;
        }
        QString idx = contents.at(0);
        QString score = contents.at(1);
        QStringList summmary = Stemmer::stemSentence(contents.at(2));
        QStringList review = Stemmer::stemSentence(contents.at(3));

        summmary.append(review);
        review.clear();

        QString stems = summmary.join(" ");
        summmary.clear();

        if (stems.size() < 100)
            continue;

        outp->write(QString("%1\t%2\t%3\n").arg(idx).arg(score).arg(stems).toUtf8());
    }
    inp->close();
    outp->close();
    delete inp;
    delete outp;

    qDebug("Done");
    return 0;

    return a.exec();
}
