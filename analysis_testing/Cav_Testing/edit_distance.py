# edit_distance.py - Take in two strings, return edit (ins/sub/del) distance
#
# Ben Cavanagh
# 09-03-2025
# Description: Take in two strings, return edit (ins/sub/del) distance
#              Based on COSC 31 pseudocode (DeepC)

def edit_distance(s, t) :
    s = str(s); t = str(t)
    m = len(s); n = len(t)
    E = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1) :
        E[i][0] = i
    for j in range(n+1) :
        E[0][j] = j

    for i in range(1, m+1) :
        for j in range(1, n+1) :
            X = 0 if s[i-1] == t[j-1] else 1
            E[i][j] = min(E[i-1][j] + 1, E[i][j-1] + 1, E[i-1][j-1] + X)

    return(E[m][n])

if __name__ == "__main__" :
    print(edit_distance("apple", "apple"))
    print(edit_distance("apple", "banana"))
    print(edit_distance("apple", "rallies"))

