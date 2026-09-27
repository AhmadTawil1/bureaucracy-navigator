from app.privacy import mask, valid_il_id


def test_valid_il_id():
    assert valid_il_id("000000018")
    assert not valid_il_id("000000019")


def test_masks_id_phone_and_email():
    text = "ישראל ישראלי, ת.ז. 000000018, טלפון 054-1234567, דוא\"ל israel@example.com"
    masked = mask(text)
    assert "000000018" not in masked and "[ת״ז]" in masked
    assert "054-1234567" not in masked and "[טלפון]" in masked
    assert "israel@example.com" not in masked and "[אימייל]" in masked


def test_numbers_glued_to_hebrew_are_masked():
    # How Word's PDF export came out in Task 3.8.
    masked = mask(".ת.ז000000018\n:טלפון054-1234567")
    assert "000000018" not in masked and "054-1234567" not in masked


def test_longer_number_is_not_masked_as_id():
    assert mask("אסמכתא 1000000018") == "אסמכתא 1000000018"


def test_invoice_number_failing_check_digit_is_kept():
    assert mask("חשבונית מספר 123456789") == "חשבונית מספר 123456789"
    assert not valid_il_id("123456789")
