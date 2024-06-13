"""
Jefferson Disk

Advantages:
1. Easy to implement and understand.
2. Provides a good level of encryption for short messages.
3. Flexible as the number of disks can be adjusted based on message length.

Disadvantages:
1. Not secure against modern cryptographic analysis.
2. Requires both parties to have the same disk configuration for decryption.
3. Disk generation must be truly random to ensure security.
"""

import string
import random
from typing import List, Tuple


class JeffersonDisk:
    def __init__(self):
        self.disks = []

    def _generate_disks(self, num_disks: int) -> List[List[str]]:
        """
        Generate a list of shuffled disks based of the length of the input message.

        One character in the input message corresponds to one shuffed disk.
        Each disk is a list of uppercase English alphabet letters in random order.

        Args:
            num_disks (int): Number of disks to generate

        Returns:
            List[List[str]] :A list of disks, where each disk is a list of shuffled uppercase letters.
        """
        self.disks = []
        for _ in range(num_disks):
            alphabet = list(string.ascii_uppercase)
            random.shuffle(alphabet)
            self.disks.append(alphabet)
        return self.disks

    def encrypt(self, text: str) -> Tuple[str, int, List[List[str]]]:
        """
        Encrypts a given text using the Jefferson Disk.

        Args:
            text (str): Input text to encrypt

        Returns
             tuple: A tuple containing the encrypted text (str), the selected row (int), and the disk configuration (List[List[str]]) used for the encryption.

        Example:
            >>> jefferson = JeffersonDisk()
            >>> message = "HELLO WORLD"
            >>> encrypted_message, selected_row = jefferson.encrypt(message)
        """
        text = text.upper().replace(" ", "")
        num_disks = len(text)
        disks = self._generate_disks(num_disks)

        selected_row = random.randint(0, 25)
        ciphertext = []

        for i, char in enumerate(text):
            # Select the char index from the i-th disk
            char_index = self.disks[i].index(char)
            # Generate the cipher character according shifts indicated by selected_row
            cipher_char = self.disks[i][(char_index + selected_row) % 26]
            ciphertext.append(cipher_char)
        return "".join(ciphertext), selected_row, disks

    def decrypt(self, text: str, selected_row: int, disks: List[List[str]]) -> str:
        """
        Decrypts a given text using the Jefferson Disk.

        Args:
            text (str): Input text to decrypt
            selected_row (int): The row number used during encryption

        Returns:
            str: Decrypted text

        Example:
            >>> jefferson = JeffersonDisk()
            >>> encrypted_message, selected_row = jefferson.encrypt("HELLO WORLD")
            >>> decrypted_message = jefferson.decrypt(encrypted_message, selected_row)
        """
        text = text.upper().replace(" ", "")
        num_disks = len(text)

        deciphertext = []

        for i, char in enumerate(text):
            # Select the char index from the i-th disk
            char_index = disks[i].index(char)
            # Generate the plain character according to shifts indicated by selected_row
            deciphertext_char = disks[i][(char_index - selected_row) % 26]
            deciphertext.append(deciphertext_char)
        return "".join(deciphertext)


# Example Usage
jefferson = JeffersonDisk()
text = "Shes a pretty girl"
encrypted_text, selected_row, disks = jefferson.encrypt(text)
decrypted_text = jefferson.decrypt(encrypted_text, selected_row, disks)

print(f"Original Message: {text}")
print(f"Encrypted Message: {encrypted_text}")
print(f"Decrypted Message: {decrypted_text}")
