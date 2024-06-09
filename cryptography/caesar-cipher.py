"""

Algorithm for Caesar Cipher

- Ancient cryptography used by Julius Caesar.
- Substitution Cipher Family
- Based on transposition of alphabets
- Encrypt and decrypt based on the shift by transposition

Advantages
- Efficient, linear time complexity of O(n)

Disadvantages
- Easily broken (only 25 possible shifts)

"""


def caesar_cipher_encrypt(text: str, shift: int) -> str:
    """
    For a given input text, encrypts it using the Caesar cipher.

    Args:
        text (str): Text to encrypt
        shift (int): Integer to shift the the alphabet by

    Returns:
        result (str): Encrypted text
    """
    result = ""

    for char in text:
        if char.isupper():
            # "A" is 65 in ASCII
            # Note modulo %26 is for the wrapping around the alphabet in the event the character + shift exceeds the 26 range of the alphabet.
            result += chr((ord(char) + shift - 65) % 26 + 65)
        elif char.islower():
            # "a" is 97 in ASCII
            result += chr((ord(char) + shift - 97) % 26 + 97)
        # not alphabetic character
        else:
            result += char

    return result


def caesar_cipher_decrypt(encrpyted_text: str, shift: int) -> str:
    """
    For a given encrypted text, dencrypts it using the Caesar cipher.

    Args:
        encrpyted_text (str): Text to dencrypt
        shift (int): Integer to shift the the alphabet back by

    Returns:
        result (str): Decrypted text
    """
    result = ""

    for char in encrpyted_text:
        if char.isupper():
            # "A" is 65 in ASCII
            result += chr((ord(char) - shift - 65) % 26 + 65)
        elif char.islower():
            # "a" is 97 in ASCII
            result += chr((ord(char) - shift - 97) % 26 + 97)
        # not alphabetic character
        else:
            result += char

    return result


# Example
text = "That girl is really pretty!"
shift = 5
encrypted_text = caesar_cipher_encrypt(text, shift)


print("Original Text: " + text)
print("Encrypted Text: " + encrypted_text)
print("Decrypted Text: " + caesar_cipher_decrypt(encrypted_text, shift))
