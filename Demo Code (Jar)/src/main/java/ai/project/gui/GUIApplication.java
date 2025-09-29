package ai.project.gui;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class GUIApplication extends Application {

	/* Run from a different class (Main.java) */
	public static void main(String[] args) {
		launch(args);
	}

	@Override
	public void start(Stage primaryStage) throws Exception {
		FXMLLoader mainLoader = ResourceManager.getFXMLLoader("main");
		Parent mainRoot = mainLoader.load();

		Scene scene = new Scene(mainRoot);

		scene.getStylesheets().add(ResourceManager.getStylesheetURL("root"));
		scene.getStylesheets().add(ResourceManager.getStylesheetURL("normal_button"));

		primaryStage.setScene(scene);
		primaryStage.setMaximized(false);
		primaryStage.setMinHeight(563);
		primaryStage.setMinWidth(700);
		primaryStage.setTitle("Arabic Search Query Auto-complete");

		primaryStage.show();
	}
}
