from collections import defaultdict
class Solution:
    def groupAnagrams(self, strs):
        hashmap = defaultdict(list)

        for word in strs:
            sorted_word = ''.join(sorted(word))
            hashmap[sorted_word].append(word)

        return list(hashmap.values())
