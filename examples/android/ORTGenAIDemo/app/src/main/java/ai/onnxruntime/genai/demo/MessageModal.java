package ai.onnxruntime.genai.demo;

public class MessageModal {

    // string to store our message and sender
    private String message;
    private String sender;

    // constructor.
    public MessageModal(String message, String sender) {
        this.message = message;
        this.sender = sender;
    }

    // getter and setter methods.
    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public String getSender() {
        return sender;
    }

    public void setSender(String sender) {
        this.sender = sender;
    }
}
