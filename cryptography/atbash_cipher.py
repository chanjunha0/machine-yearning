"""
Algorithm for Atbash Cipher

- Ancient cryptography originally used for the Hebrew alphabet.
- Substitution Cipher Family
- Based on reversing the alphabet
- Encrypt and decrypt by mapping each letter to its reverse

Advantages
- Simple and easy to implement
- Symmetric encryption and decryption process

Disadvantages
- Easily broken (provides no real cryptographic security)

"""


def atbash_cipher(text: str) -> str:
    """
    Encrypts or decrypts a given input using the Atbash Cipher.

    Args:
        text (str): The text to encrypt or decrypt.

    Returns:
        str: The encrypted or decrypted text.

    """
    # Define the alphabet
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Create the reversed alphabet
    reversed_alphabet = alphabet[::-1]

    # Create translation table for mapping uppercase letters
    translation_table_upper = str.maketrans(alphabet, reversed_alphabet)

    # Create translation table for mapping lowercase letters
    translation_table_lower = str.maketrans(alphabet.lower(), reversed_alphabet.lower())

    # Translate the text using the tables
    translated_text = text.translate(translation_table_upper).translate(
        translation_table_lower
    )

    return translated_text


# Example Use
text = "That girl is really pretty!"
encrypted_text = atbash_cipher(text)
decrypted_text = atbash_cipher(encrypted_text)

print("Original Text: " + text)
print("Encrypted Text: " + encrypted_text)
print("Decrypted Text: " + decrypted_text)
