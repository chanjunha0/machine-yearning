from typing import List, Union


def boyer_moore_majority_vote(nums: List[int]) -> Union[int, None]:
    """
    Implements the Boyer-Moore Majority Voting Algorithm to find the majority element in a list.

    The majority element is an element that appears more than n/2 times in the list,
    where n is the length of the list.

    Args:
        nums (List[int]): A list of integers.

    Returns:
        Union[int, None]: The majority element if it exists, None otherwise.

    Time complexity: O(n)
    Space complexity: O(1)
    """
    candidate = None
    count = 0

    for num in nums:
        if count == 0:
            candidate = num
        count += 1 if num == candidate else -1

    # Verify if the candidate is actually the majority element
    if nums.count(candidate) > len(nums) // 2:
        return candidate
    return None
