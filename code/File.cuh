//convert from relative path into absolute path
#pragma once
#include <string>
#include "direct.h"

using namespace std;

constexpr int MAXLENGTH = 1000;

static string getCurrentWorkingDirectory() 
{
    char cwd[MAXLENGTH];
    auto _ = _getcwd(cwd, MAXLENGTH);
    return string(cwd);
}

static string getParentDirectory(const string& path) 
{
    size_t pos1 = path.find_last_of("/");
    size_t pos2 = path.find_last_of("\\");

    if (pos1 != string::npos)
        return path.substr(0, pos1);
    if (pos2 != string::npos)
        return path.substr(0, pos2);
    return "";
}

static string getLastPiece(const string& path)
{
    size_t pos1 = path.find_last_of("/");
    size_t pos2 = path.find_last_of("\\");

    if (pos1 != string::npos)
        return path.substr(pos1+1, path.length()-pos1);
    if (pos2 != string::npos)
        return path.substr(pos2+1, path.length()-pos1);
    return "";
}

static string getProjectDirectory()
{
    string cwd = getCurrentWorkingDirectory();

    while (true)
    {
        if (getLastPiece(cwd) == "Graphics(CUDA)")
            return cwd;

        cwd = getParentDirectory(cwd);
        if (cwd.empty())
            return "";
    }
}

static string getInputFilePath(const char* fileName)
{
    string projectDir = getProjectDirectory();
    if (projectDir.empty())
        return "<file not found>";
    else
        return projectDir + "/input/" + string(fileName);
}

static string getOutputFilePath(const char* fileName)
{
    string projectDir = getProjectDirectory();
    if (projectDir.empty())
        return "<file not found>";
    else
        return projectDir + "/output/" + string(fileName);
}
