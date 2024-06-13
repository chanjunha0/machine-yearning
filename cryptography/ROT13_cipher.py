"""
ROT13 Cipher

Advantages
1. Simple to implement and use.
2. Reversible without the need for a key, as applying ROT13 twice returns the original text.
3. Preserves letter case and punctuation.

Disadvantages
1. Provides minimal security and can be easily broken with basic knowledge of the cipher.
2. Not suitable for encrypting sensitive or important information.
3. Limited to alphabetic characters and does not encrypt numbers or special characters.
"""


def rot13_cipher(text):
    """
    Encrypts or decrpyts a given text using the ROT13 cipher

    Replaces each letter with the letter 13 positions after it in the alphabet. Encryption and decryption uses the same algorithm (alphabet has 26 letters)

    Args:
        text (str): Encrypted or decrypted text

    Return:
        str: The encrypted or decrypted text.

    Example:
        >>> rot13_cipher("Hello, World!")
        'Uryyb, Jbeyq!'

        >>> rot13_cipher("Uryyb, Jbeyq!")
        'Hello, World!'
    """
    results = []
    for letter in text:
        if letter.isupper():
            results += chr((ord(letter) - 65 + 13) % 26 + 65)
        elif letter.islower():
            results += chr((ord(letter) - 97 + 13) % 26 + 97)
        else:
            results.append(letter)
    return "".join(results)


# Example Use
text = "That's a pretty girl"
encrypted_text = rot13_cipher(text)
decrypted_text = rot13_cipher(encrypted_text)

print("Original text:", text)
print("Encoded text:", encrypted_text)
print("Decoded text:", decrypted_text)
