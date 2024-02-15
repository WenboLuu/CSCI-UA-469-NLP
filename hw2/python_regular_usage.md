In Python's `re` module, `re.search`, `re.match`, and `re.findall` are three methods used to search text for matches, but they differ in how they search and what they return. Here's a comparison in a markdown table format:

| Feature/Method | `re.search` | `re.match` | `re.findall` |
|----------------|-------------|------------|--------------|
| **Search Scope** | Searches for the first occurrence of the pattern anywhere in the string. | Searches only at the beginning of the string. | Searches for all occurrences of the pattern anywhere in the string and returns all non-overlapping matches. |
| **Return Value** | Returns a match object of the first match. If there is no match, returns `None`. | Returns a match object if the pattern is found at the beginning of the string. If not found, returns `None`. | Returns a list of strings containing all non-overlapping matches of the pattern. If groups are present in the pattern, returns a list of groups. |
| **Use Case** | Use when you need to find a pattern anywhere in the string and are interested in the first occurrence. | Use when you need to check if the string starts with a specific pattern. | Use when you need to find all occurrences of a pattern in a string and are not interested in match objects. |
| **Example** | `re.search('pattern', 'text')` | `re.match('pattern', 'text')` | `re.findall('pattern', 'text')` |

### Example Usage and Results:

Assuming the string `"Example text with pattern in the middle and pattern at the end."` and the pattern `'pattern'`:

- `re.search('pattern', 'Example text...')` would return a match object corresponding to the first occurrence of "pattern".
  
- `re.match('pattern', 'Example text...')` would return `None` because "pattern" is not at the beginning of the string.
  
- `re.findall('pattern', 'Example text...')` would return `['pattern', 'pattern']`, a list containing all occurrences of "pattern".

This comparison highlights the differences in application and output of these three commonly used methods from Python's `re` module.
